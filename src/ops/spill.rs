//! Heap-or-mmap spill buffer for size-known-upfront allocations.
//!
//! Several `Result`-returning public APIs in `dia` allocate flat
//! scratch buffers proportional to input size: AHC pdist
//! (`n*(n-1)/2` f64 cells), reconstruct grids, count-tensor
//! aggregates. Past a few hundred MB these can OOM-abort the
//! process from a `Result`-returning API.
//!
//! ## Two types: write-phase and read-phase
//!
//! Inspired by [`bytes::BytesMut`] / [`bytes::Bytes`]:
//!
//! - [`SpillBytesMut<T>`] — **write-phase**, unique ownership. Use
//!   while filling the buffer (`as_mut_slice`). Picks heap or
//!   file-backed mmap at construction based on
//!   [`SpillOptions::threshold_bytes`].
//! - [`SpillBytes<T>`] — **read-phase**, cheap `Clone` (`Arc`-wrapped
//!   on both backends). `Send + Sync`. Use to fan out a fully-built
//!   buffer to multiple downstream consumers.
//!
//! Convert with [`SpillBytesMut::freeze`]:
//!
//! ```ignore
//! use diarization::ops::spill::{SpillBytesMut, SpillOptions};
//! let opts = SpillOptions::default();
//! let mut buf: SpillBytesMut<f64> = SpillBytesMut::zeros(1024, &opts).unwrap();
//! for (i, slot) in buf.as_mut_slice().iter_mut().enumerate() {
//!     *slot = i as f64;
//! }
//! let frozen = buf.freeze();
//! let a = frozen.clone(); // O(1): bumps the Arc refcount.
//! let b = frozen.clone(); // O(1).
//! assert_eq!(a.as_slice(), b.as_slice());
//! ```
//!
//! ### Why two types
//!
//! - **Write phase** wants `&mut [T]` access for in-place fill —
//!   unique ownership. Cheap clone is irrelevant here: the buffer
//!   doesn't exist yet from the consumer's perspective.
//! - **Read phase** is the natural place for fan-out — the buffer
//!   is fully built and downstream may want multiple readers
//!   (different threads, different consumers). `Arc` gives O(1)
//!   `Clone` and `Send + Sync` without copying the underlying data.
//!
//! Once frozen, [`SpillBytes`] cannot be mutated; the type system
//! enforces this (no `as_mut_slice`). `freeze` is zero-copy on the
//! mmap backend (the `Arc::new` wraps the existing mapping) and
//! zero-copy on the heap backend (the `Arc<[T]>` is allocated up
//! front in [`SpillBytesMut::zeros`] with refcount 1, and `freeze`
//! moves it out unchanged).
//!
//! ## Backends
//!
//! - **Heap** (`Arc<[T]>` with refcount 1 in `SpillBytesMut`) when
//!   the requested allocation fits under
//!   [`SpillOptions::threshold_bytes`].
//! - **File-backed mmap** (over an unlinked tempfile) above the
//!   threshold. Pages are evicted to disk by the kernel page cache
//!   under memory pressure, keeping resident RAM bounded.
//!
//! The spill backend deliberately is NOT anonymous mmap
//! (`MAP_ANONYMOUS`): anonymous mmap stores dirty pages in
//! RAM + swap and consumes physical memory identically to `Vec`.
//! File-backed mmap is what actually trades RAM for disk.
//!
//! ### mmap backing-file safety
//!
//! The `unsafe MmapOptions::map_mut` call requires that the
//! underlying file not be modified concurrently by another writer.
//! We obtain that guarantee differently per platform:
//!
//! - **Linux / Android**: `open(dir, O_TMPFILE | O_RDWR, 0600)`
//!   creates an anonymous inode that has *never* been linked
//!   into the directory. No path on disk; no race window. If the
//!   filesystem does not support `O_TMPFILE` (NFS, some FUSE, very
//!   old kernels), the syscall fails with `EOPNOTSUPP` / `EISDIR`
//!   and we surface it as `SpillError::TempfileCreation` rather
//!   than silently falling back. Configure
//!   [`SpillOptions::with_spill_dir`] to point at an
//!   `O_TMPFILE`-supporting filesystem (ext4 / xfs / btrfs / tmpfs)
//!   if your default `/tmp` is on one that does not.
//!
//! - **macOS / other Unix**: no `O_TMPFILE` equivalent. We fall
//!   back to `tempfile::tempfile[_in]`, which uses `mkstemp + unlink`
//!   — a microsecond-scale race window where the random 0600 path
//!   is briefly visible. After unlink, `nlink() == 0` is verified
//!   defensively (`SpillError::TempfileNotUnlinked` if not), but
//!   that check cannot retroactively close the create-window race.
//!   This residual exposure is acceptable for **single-tenant
//!   container** deployments (the dominant target) but should be
//!   considered when running on a shared-UID multi-tenant host;
//!   such deployments should prefer Linux with O_TMPFILE-supporting
//!   storage.
//!
//! - **Windows**: `FILE_FLAG_DELETE_ON_CLOSE` with sharing denied
//!   (via `tempfile`); no other process can open the file at all.
//!
//! ## Transparent Huge Pages (Linux)
//!
//! On Linux, mmap'd buffers are advised with `MADV_HUGEPAGE`
//! (`Advice::HugePage`), which lets the kernel back the mapping with
//! 2 MiB pages where the THP policy permits. Reduces TLB pressure
//! on the dominant access patterns (sequential read of pdist /
//! aggregate buffers). The advise is opportunistic — silently
//! degrades to regular 4 KiB pages on kernels with THP disabled,
//! older kernels, or filesystems that don't support it.
//!
//! `memmapix` also exposes `MmapOptions::huge(Some(N))` which sets
//! `MAP_HUGETLB` on the resulting mapping, but only for `map_anon`:
//! the `map_mut` codepath ignores the `huge` field. We use
//! `map_mut(&tempfile)` (file-backed; spills dirty pages to disk)
//! rather than `map_anon` (anonymous; dirty pages stay in
//! RAM + swap), so `huge()` is unreachable for our backend.
//! Reaching `MAP_HUGETLB` over a tempfile would also require
//! mounting the file's parent on `hugetlbfs` plus a preconfigured
//! kernel hugepage pool — the wrong tradeoff for an opportunistic
//! perf hint that should fail soft. `MADV_HUGEPAGE` covers the
//! same TLB-win territory without those constraints.
//!
//! ## Configuration
//!
//! [`SpillBytesMut::zeros`] takes the [`SpillOptions`] explicitly as
//! a `&SpillOptions` argument — no process-global, no thread-local,
//! no action-at-distance. Each top-level Options struct in `dia`
//! (`OwnedPipelineOptions`, `OfflineInput`, `AssignEmbeddingsInput`,
//! `ReconstructInput`, `StreamingOfflineOptions`) carries a
//! [`SpillOptions`] field defaulting to [`SpillOptions::default`];
//! the corresponding entry function passes a borrow of that field
//! down to every transitive `SpillBytesMut::zeros` call site.
//! Concurrent multi-threaded calls cannot interfere because there
//! is no shared mutable state.
//!
//! Default: 64 MiB threshold, [`std::env::temp_dir`] for the spill
//! file. Production deployments where `/tmp` is `tmpfs` (Docker
//! default) **must** override `spill_dir` to a real-disk path,
//! otherwise "spill to disk" is a misnomer and the OOM concern
//! still applies.
//!
//! ## Type contract
//!
//! Both [`SpillBytesMut<T>`] and [`SpillBytes<T>`] require
//! `T: bytemuck::Pod` — the type must be plain-old-data (no padding,
//! no destructors, every byte pattern valid). `f64`, `f32`, `u8`,
//! `u16`, `u32`, `u64`, `usize`, signed variants all qualify; `bool`
//! does NOT (only `0u8` and `1u8` are valid). Mask buffers that
//! previously stored `Vec<bool>` migrate to `Vec<u8>` (0/1) when
//! wrapped in `SpillBytesMut<u8>`.
//!
//! ## Limitations
//!
//! - Sized once at construction. No `push`/`grow`. Every call site
//!   in `dia` knows the buffer length upfront, so this is fine.
//! - [`SpillBytesMut`] is `Send` but not `Sync`: `as_mut_slice`
//!   exposes `&mut [T]` whose aliasing semantics require unique
//!   access.
//! - [`SpillBytes`] is `Send + Sync`: read-only access is safe to
//!   share across threads.

// Internal call sites currently use `as_mut_slice` exclusively;
// the read-only / inspection accessors and the configuration
// setters are part of the public API for downstream consumers and
// tests, but Rust flags them as "never used" inside the crate.
#![allow(dead_code)]

use core::marker::PhantomData;
use std::{
  path::{Path, PathBuf},
  sync::Arc,
};

use bytemuck::Pod;
#[cfg(target_os = "linux")]
use memmapix::Advice;
#[cfg(any(unix, windows))]
use memmapix::{MmapMut, MmapOptions};

/// Errors returned by [`SpillBytesMut`] allocation.
#[derive(Debug, thiserror::Error)]
pub enum SpillError {
  /// `n.checked_mul(size_of::<T>())` overflowed `usize`. The caller
  /// asked for an allocation past `usize::MAX` bytes.
  #[error("spill: requested element count {n} times size_of::<T>={element_size} overflows usize")]
  SizeOverflow {
    /// Requested element count.
    n: usize,
    /// Per-element size (`size_of::<T>()`).
    element_size: usize,
  },
  /// Failed to create the unlinked tempfile that backs the mmap.
  /// Realistic causes: `ENOSPC`, `EACCES`, `EROFS`, missing
  /// `spill_dir` permissions.
  #[error("spill: failed to create tempfile in {dir:?}: {source}")]
  TempfileCreation {
    /// Directory the tempfile was created in (`None` =
    /// [`std::env::temp_dir`]).
    dir: Option<PathBuf>,
    /// Underlying I/O error.
    #[source]
    source: std::io::Error,
  },
  /// Failed to grow the tempfile to the requested size via
  /// `set_len`. Typically `ENOSPC`.
  #[error("spill: failed to grow tempfile to {bytes} bytes: {source}")]
  TempfileGrow {
    /// Requested file length in bytes.
    bytes: u64,
    /// Underlying I/O error.
    #[source]
    source: std::io::Error,
  },
  /// `mmap()` failed. Realistic causes on Linux: `EAGAIN` (locked
  /// memory limit), `ENFILE`/`EMFILE` (fd limit), `ENOMEM`
  /// (kernel-side address-space exhaustion).
  #[error("spill: mmap failed for {bytes} bytes: {source}")]
  MmapFailed {
    /// Requested mapping length in bytes.
    bytes: usize,
    /// Underlying I/O error from the mmap syscall.
    #[source]
    source: std::io::Error,
  },
  /// `tempfile::tempfile[_in]` returned a file with a non-zero
  /// link count, so the backing file is still reachable by name
  /// from the spill directory. This happens when the underlying
  /// filesystem (e.g. NFS, or an old Linux without `O_TMPFILE`)
  /// makes the unlink-on-create fast path unavailable; the
  /// `tempfile` 3.x fallback creates a named file and ignores
  /// `remove_file` failures, leaving the file linked.
  ///
  /// We refuse to map a still-linked file because it violates the
  /// `unsafe MmapOptions::map_mut` precondition: another same-UID
  /// process could open and modify the file behind our back,
  /// breaking the read-only invariant of `SpillBytes` after
  /// `freeze`. Configure a `spill_dir` on a filesystem that
  /// supports unlinked tempfiles to avoid this.
  #[error(
    "spill: tempfile in {dir:?} was not unlinked at creation \
     (filesystem does not support O_TMPFILE-style unlink-private \
     temp files); refusing to map writable buffer that other \
     same-UID processes can still open by path"
  )]
  TempfileNotUnlinked {
    /// Directory the tempfile was created in (`None` =
    /// [`std::env::temp_dir`]).
    dir: Option<PathBuf>,
  },
  /// `posix_fallocate(2)` failed to reserve disk blocks for the
  /// mmap-backed tempfile. Without preallocation, `set_len` alone
  /// produces a sparse file whose pages may be backed only after
  /// the kernel observes a write fault; running out of disk space
  /// at fault time is delivered as `SIGBUS` (process crash) rather
  /// than as a typed I/O error from a syscall. We reserve up front
  /// so the spill backend either succeeds with a fully-backed file
  /// or returns this error.
  #[error("spill: failed to preallocate {bytes} bytes for tempfile: {source}")]
  TempfilePreallocate {
    /// Requested file length in bytes.
    bytes: u64,
    /// Underlying I/O error.
    #[source]
    source: std::io::Error,
  },
  /// The host target does not support file-backed spilling. The
  /// mmap path requires `cfg(any(unix, windows))` because it leans
  /// on `fs4::FileExt` (`posix_fallocate` / `F_PREALLOCATE` /
  /// `SetFileValidData`) and tempfile semantics. wasm32 / WASI / etc.
  /// build the lib but the spill mmap path is compiled out, so an
  /// allocation above [`SpillOptions::threshold_bytes`] surfaces this
  /// variant. Callers can either lower the input size, raise the
  /// threshold above the requested allocation, or treat this as a
  /// hard fail on the unsupported target.
  #[error(
    "spill: host target does not support file-backed spilling \
     (allocation of {bytes} bytes exceeds the heap threshold but \
     this build was compiled without the unix/windows mmap path)"
  )]
  UnsupportedTarget {
    /// Requested allocation in bytes.
    bytes: u64,
  },
}

#[cfg_attr(not(tarpaulin), inline(always))]
const fn default_threshold_bytes() -> usize {
  SpillOptions::DEFAULT_THRESHOLD_BYTES
}

/// Configuration for the spill backend. All fields are private;
/// access via the getters and modify via the `with_*` / `set_*`
/// builders.
///
/// Construct via [`SpillOptions::new`] (`const fn`) or [`Default`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpillOptions {
  threshold_bytes: usize,
  #[cfg_attr(
    feature = "serde",
    serde(skip_serializing_if = "Option::is_none", default)
  )]
  spill_dir: Option<PathBuf>,
}

impl SpillOptions {
  /// Default threshold: 64 MiB. Allocations smaller than this stay
  /// on the heap; larger ones spill to file-backed mmap.
  ///
  /// 64 MiB is a defensive choice for the containerized inference
  /// workloads `dia` typically runs in: 1–2 GiB total memory
  /// budget, model weights + ORT/torch runtime + audio buffers
  /// already consuming several hundred MB, and **multiple**
  /// `SpillBytesMut` allocations live concurrently on a single
  /// pipeline call (segmentations + raw_embeddings + count×2 + AHC
  /// pdist + reconstruct grids ×4). A higher threshold (e.g.
  /// 256 MiB) lets each individual allocation pass the cap while
  /// the aggregate quietly stacks into multi-GB heap usage and
  /// OOMs the container.
  ///
  /// At 64 MiB:
  /// - typical sub-hour pipeline calls stay heap-resident (the
  ///   per-buffer ceiling for 1 h of audio is ~50 MB);
  /// - multi-hour batches and adversarial inputs spill earlier,
  ///   well before they can stack into an OOM;
  /// - the page-fault cost on workloads that would have fit on
  ///   heap is sub-millisecond per page on NVMe — negligible
  ///   compared to an OOM crash.
  ///
  /// Override per-call via [`SpillOptions::with_threshold_bytes`]
  /// when the deployment has a known different memory profile.
  pub const DEFAULT_THRESHOLD_BYTES: usize = 64 * 1024 * 1024;

  /// Construct with default values: 64 MiB threshold,
  /// [`std::env::temp_dir`] for the spill directory.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      threshold_bytes: default_threshold_bytes(),
      spill_dir: None,
    }
  }

  /// Threshold (bytes) above which an allocation spills to mmap.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn threshold_bytes(&self) -> usize {
    self.threshold_bytes
  }

  /// Spill directory. `None` ⇒ [`std::env::temp_dir`]. Override to a
  /// real-disk path on deployments where `/tmp` is `tmpfs` (Docker
  /// default) — otherwise spilled pages live in RAM-backed `tmpfs`
  /// and the OOM concern is unaddressed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn spill_dir(&self) -> Option<&Path> {
    self.spill_dir.as_deref()
  }

  /// Builder: set the spill threshold.
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_threshold_bytes(mut self, threshold_bytes: usize) -> Self {
    self.set_threshold_bytes(threshold_bytes);
    self
  }

  /// Builder: set the spill directory. `None` resets to
  /// [`std::env::temp_dir`].
  #[must_use]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_spill_dir(mut self, spill_dir: Option<PathBuf>) -> Self {
    self.set_spill_dir(spill_dir);
    self
  }

  /// Mutating: set the spill threshold.
  ///
  /// `const fn` because `usize` has no destructor; the
  /// `with_threshold_bytes` builder is `const` and forwards here.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_threshold_bytes(&mut self, threshold_bytes: usize) -> &mut Self {
    self.threshold_bytes = threshold_bytes;
    self
  }

  /// Mutating: set the spill directory.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_spill_dir(&mut self, spill_dir: Option<PathBuf>) -> &mut Self {
    self.spill_dir = spill_dir;
    self
  }
}

impl Default for SpillOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

// ── Mmap backing handle ───────────────────────────────────────────

/// Inner mmap state shared between [`SpillBytesMut`] (during the
/// write phase) and [`SpillBytes`] (after `freeze`). Holds the
/// mapping plus the unlinked tempfile that backs it; both are
/// dropped together when the last `Arc<MmapHandle>` goes away.
///
/// File-backed spilling requires platform APIs (`O_TMPFILE` /
/// `FILE_FLAG_DELETE_ON_CLOSE`, `posix_fallocate` / `F_PREALLOCATE`
/// / `SetFileValidData`, mmap) that `fs4` and `memmapix` only
/// implement on `cfg(any(unix, windows))`. On other targets
/// (wasm32, WASI, …) this struct and the surrounding mmap path are
/// compiled out; an above-threshold allocation surfaces
/// [`SpillError::UnsupportedTarget`] instead.
#[cfg(any(unix, windows))]
struct MmapHandle {
  /// We keep `MmapMut` even after freeze; the type-system
  /// invariant is that `SpillBytes` only ever borrows it through
  /// `&MmapHandle` (no `&mut`), so no mutation is reachable. We do
  /// not call `make_read_only` (which would `mprotect` to
  /// `PROT_READ`) because the syscall is unnecessary for Rust's
  /// type-level enforcement and adds a failure mode.
  map: MmapMut,
  /// Unlinked anonymous file owning the on-disk storage. Created
  /// via `tempfile::tempfile[_in]`, which on Unix unlinks the file
  /// from the directory immediately so no path is visible to other
  /// processes; on Windows it uses `FILE_FLAG_DELETE_ON_CLOSE` with
  /// share-deny set. Either way, no same-UID process can open the
  /// file by path while we hold the handle, which is the precondition
  /// the `unsafe` `MmapOptions::map_mut` call relies on.
  _file: std::fs::File,
}

// ── SpillBytesMut: write-phase, unique ownership ──────────────────

/// A fixed-size flat buffer that picks heap-or-mmap at construction
/// time based on the [`SpillOptions`] passed to [`Self::zeros`].
///
/// Use during the **write phase**: fill via `as_mut_slice`. Convert
/// to [`SpillBytes`] via [`Self::freeze`] when ready to publish for
/// fan-out.
///
/// `T: Pod` so the byte buffer underlying the mmap can be
/// reinterpreted as `&[T]` / `&mut [T]` without UB. `bool` is NOT
/// `Pod` (only `0u8` and `1u8` are valid byte patterns); use
/// `Vec<u8>`-as-mask wrapped in `SpillBytesMut<u8>` for boolean
/// masks.
pub struct SpillBytesMut<T> {
  inner: SpillMutInner<T>,
  len: usize,
  _phantom: PhantomData<T>,
}

enum SpillMutInner<T> {
  /// Unique-refcount `Arc<[T]>` so that `freeze` can hand the same
  /// allocation to [`SpillBytes::Heap`] without a copy. We never
  /// clone the inner Arc, so `Arc::get_mut` always succeeds.
  Heap(Arc<[T]>),
  /// `_file` owns the unlinked tempfile so its lifetime ≥ the
  /// mmap's. We use `tempfile::tempfile[_in]` which returns a
  /// `std::fs::File` that has already been unlinked from the
  /// directory (Unix) or opened with `FILE_FLAG_DELETE_ON_CLOSE`
  /// (Windows); no path is visible to other processes while the
  /// mapping is live. Dropping the file reclaims the disk space.
  ///
  /// Compiled out on `cfg(not(any(unix, windows)))`; see
  /// [`MmapHandle`] / [`SpillError::UnsupportedTarget`].
  #[cfg(any(unix, windows))]
  Mmap { map: MmapMut, _file: std::fs::File },
}

/// Open the unlinked file that backs an mmap-spilled `SpillBytesMut`.
///
/// On Linux/Android we call `open(dir, O_TMPFILE | O_RDWR, 0o600)`
/// directly via `libc` so the file is anonymous from creation —
/// there is no path on disk for another process to find, no race
/// window between create and unlink. If the filesystem does not
/// support `O_TMPFILE` (rare in modern container deployments;
/// NFS / some FUSE / very old kernels) the syscall returns
/// `EOPNOTSUPP` / `EISDIR` and we surface it as
/// `TempfileCreation`. Production deployments with such storage
/// should configure `SpillOptions::with_spill_dir` to point at an
/// `O_TMPFILE`-supporting filesystem (ext4 / xfs / btrfs / tmpfs)
/// or ensure the spill backend is never reached.
///
/// On other Unix (macOS, BSDs) and Windows we fall back to
/// `tempfile::tempfile[_in]`. macOS has no `O_TMPFILE` analogue;
/// the `mkstemp + unlink` race window is inherent to POSIX. Windows
/// uses `FILE_FLAG_DELETE_ON_CLOSE` with sharing denied, which
/// prevents external opens entirely.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn open_backing_file(spill_dir: Option<&Path>) -> Result<std::fs::File, SpillError> {
  use rustix::fs::{Mode, OFlags};
  // `O_TMPFILE` is not exposed on stable `std::fs`; rustix wraps
  // the syscall directly. The open target is a *directory* — the
  // kernel uses it to pick the mount point for the unnamed inode.
  // If the filesystem doesn't support `O_TMPFILE` (NFS / some FUSE
  // / very old kernels), the syscall returns `EOPNOTSUPP`/`EISDIR`
  // and we surface it as `TempfileCreation`.
  let dir_owned = match spill_dir {
    Some(d) => d.to_path_buf(),
    None => std::env::temp_dir(),
  };
  let owned_fd = rustix::fs::open(
    &dir_owned,
    OFlags::RDWR | OFlags::TMPFILE | OFlags::CLOEXEC,
    Mode::from_bits_truncate(0o600),
  )
  .map_err(|errno| SpillError::TempfileCreation {
    dir: spill_dir.map(|d| d.to_path_buf()),
    source: std::io::Error::from(errno),
  })?;
  // `OwnedFd` → `std::fs::File` is a zero-cost conversion: both
  // own the same raw fd and close it on drop. After this point the
  // file is just a regular `std::fs::File` from the rest of the
  // module's perspective.
  Ok(std::fs::File::from(owned_fd))
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn open_backing_file(spill_dir: Option<&Path>) -> Result<std::fs::File, SpillError> {
  match spill_dir {
    Some(dir) => tempfile::tempfile_in(dir),
    None => tempfile::tempfile(),
  }
  .map_err(|source| SpillError::TempfileCreation {
    dir: spill_dir.map(|d| d.to_path_buf()),
    source,
  })
}

impl<T: Pod> SpillBytesMut<T> {
  /// Allocate `n` zero-initialized cells of `T` using the supplied
  /// [`SpillOptions`].
  ///
  /// Picks heap if `n * size_of::<T>() ≤ opts.threshold_bytes()`,
  /// else file-backed mmap in [`SpillOptions::spill_dir`]. Both
  /// backends return zero-initialized cells.
  ///
  /// `opts` is borrowed for the duration of the call; subsequent
  /// allocations may use a different `SpillOptions`. The resulting
  /// buffer is committed to its backend and unaffected by later
  /// changes to the caller's `SpillOptions`.
  pub fn zeros(n: usize, opts: &SpillOptions) -> Result<Self, SpillError> {
    let element_size = core::mem::size_of::<T>();
    let bytes = n
      .checked_mul(element_size)
      .ok_or(SpillError::SizeOverflow { n, element_size })?;

    // Special case: `n == 0` always returns an empty heap buffer.
    // mmap of length 0 is undefined / EINVAL on most platforms.
    if bytes == 0 {
      return Ok(Self {
        inner: SpillMutInner::Heap(Arc::from(Vec::<T>::new())),
        len: 0,
        _phantom: PhantomData,
      });
    }

    if bytes <= opts.threshold_bytes() {
      // Heap path: allocate `Arc<[T]>` directly (refcount 1, weak
      // count 1) so `freeze` is a zero-copy move. Zero-fill via
      // `T::zeroed()` (Pod requires Zeroable).
      let arc: Arc<[T]> = std::iter::repeat_n(T::zeroed(), n).collect();
      Ok(Self {
        inner: SpillMutInner::Heap(arc),
        len: n,
        _phantom: PhantomData,
      })
    } else {
      // mmap path. Only supported on `cfg(any(unix, windows))` —
      // wasm/WASI builds compile this branch out via the routing
      // below and surface `SpillError::UnsupportedTarget` instead.
      #[cfg(any(unix, windows))]
      {
        Self::new_mmap(n, bytes, opts.spill_dir())
      }
      #[cfg(not(any(unix, windows)))]
      {
        let _ = (n, opts);
        Err(SpillError::UnsupportedTarget {
          bytes: bytes as u64,
        })
      }
    }
  }

  #[cfg(any(unix, windows))]
  fn new_mmap(n: usize, bytes: usize, spill_dir: Option<&Path>) -> Result<Self, SpillError> {
    // Backing-file creation strategy depends on the platform:
    //
    // * **Linux/Android**: `open(dir, O_TMPFILE | O_RDWR, 0o600)`
    //   creates an anonymous inode that has *never* been linked
    //   into the directory. There is no race window between create
    //   and unlink for another same-UID process to grab a writable
    //   fd by path — the path simply does not exist. If the
    //   filesystem doesn't support `O_TMPFILE` (NFS, some FUSE,
    //   very old Linux), the kernel returns `EOPNOTSUPP`/`EISDIR`,
    //   and we surface that as `TempfileCreation` rather than
    //   silently falling back to `mkstemp + unlink`.
    //
    // * **macOS / other Unix**: no `O_TMPFILE` equivalent.
    //   `tempfile::tempfile_in` calls `mkstemp` then `unlink` (the
    //   classic POSIX dance) — there is a microsecond-scale race
    //   window in which the random 0600 path is visible. The
    //   subsequent `nlink() == 0` check verifies the unlink
    //   succeeded but cannot retroactively close the race; we
    //   accept that residual exposure for single-tenant container
    //   deployments and document it in the module-level docs.
    //
    // * **Windows**: `tempfile::tempfile_in` uses
    //   `FILE_FLAG_DELETE_ON_CLOSE` with sharing denied; no other
    //   process can open the file at all.
    let file = open_backing_file(spill_dir)?;
    // Defense-in-depth: refuse to map a still-linked file. On
    // Linux/Android the `O_TMPFILE` path makes this provably 0;
    // on macOS / other Unix this catches the case where `unlink`
    // failed entirely and we'd otherwise map a file an external
    // observer can still write to.
    #[cfg(unix)]
    {
      use std::os::unix::fs::MetadataExt;
      let m = file.metadata().map_err(|source| SpillError::TempfileGrow {
        bytes: bytes as u64,
        source,
      })?;
      if m.nlink() != 0 {
        return Err(SpillError::TempfileNotUnlinked {
          dir: spill_dir.map(|d| d.to_path_buf()),
        });
      }
    }
    // Reserve disk blocks before mapping. `set_len` alone produces
    // a sparse file whose pages may not have backing storage; a
    // write through the mmap that touches an unbacked page hits
    // ENOSPC as `SIGBUS` (process crash) rather than as a typed
    // I/O error. `fs4::FileExt::allocate` cross-platform-wraps
    // `posix_fallocate(2)` (Linux/Android),
    // `fcntl(F_PREALLOCATE)` (macOS), and
    // `SetFileValidData`/`SetEndOfFile` (Windows). Either we
    // succeed here with a fully-backed file or we surface
    // `SpillError::TempfilePreallocate` to the caller.
    {
      use fs4::FileExt;
      file
        .allocate(bytes as u64)
        .map_err(|source| SpillError::TempfilePreallocate {
          bytes: bytes as u64,
          source,
        })?;
    }
    // SAFETY: `file` is a freshly created, already-unlinked tempfile
    // (verified above on Unix via `nlink() == 0`). No other process
    // can open it by path; no other thread holds the handle (we own
    // it exclusively here, and only hand it out wrapped in
    // `SpillBytesMut`/`Arc<MmapHandle>` which never expose
    // `&mut File`). That satisfies `MmapOptions::map_mut`'s
    // requirement that the underlying file not be modified
    // concurrently. Disk blocks are reserved by the
    // `posix_fallocate` (Linux/Android) or `set_len` (other
    // platforms) call above, so writes through the mmap will not
    // SIGBUS on ENOSPC for the preallocated platforms.
    let map = unsafe {
      MmapOptions::new()
        .len(bytes)
        .map_mut(&file)
        .map_err(|source| SpillError::MmapFailed { bytes, source })?
    };
    // Linux: hint the kernel to back the mapping with Transparent
    // Huge Pages where possible. Reduces TLB pressure for the
    // sequential read patterns in pdist/reconstruct (64 MB+
    // mappings touch ~64k regular pages but only ~128 huge pages).
    //
    // This is a HINT — `MADV_HUGEPAGE` is silently a no-op on
    // kernels where THP is disabled (`echo never >
    // /sys/kernel/mm/transparent_hugepage/enabled`), embedded
    // builds without THP, or filesystems that don't support it.
    // We deliberately do NOT use `MAP_HUGETLB`: it requires the
    // file to live on `hugetlbfs` and hard-fails if the kernel
    // hugepage pool is empty — wrong tradeoff for an opportunistic
    // optimization.
    //
    // We ignore the error result: a failed `madvise` on a freshly
    // created mapping is benign (the mapping is still valid),
    // and we don't want a system policy decision to fail an
    // otherwise-successful allocation.
    #[cfg(target_os = "linux")]
    let _ = map.advise(Advice::HugePage);
    Ok(Self {
      inner: SpillMutInner::Mmap { map, _file: file },
      len: n,
      _phantom: PhantomData,
    })
  }

  /// Number of `T` cells in the buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn len(&self) -> usize {
    self.len
  }

  /// `true` if the buffer is empty.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// Borrow the buffer as `&[T]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn as_slice(&self) -> &[T] {
    match &self.inner {
      SpillMutInner::Heap(arc) => arc,
      #[cfg(any(unix, windows))]
      SpillMutInner::Mmap { map, .. } => {
        let bytes: &[u8] = &map[..];
        if bytes.is_empty() {
          return &[];
        }
        bytemuck::cast_slice(bytes)
      }
    }
  }

  /// Borrow the buffer as `&mut [T]`.
  ///
  /// On the heap path this is `Arc::get_mut`. We never clone the
  /// inner `Arc` while in `SpillBytesMut`, so the refcount is
  /// always 1 and `get_mut` succeeds. The `expect` is genuinely
  /// unreachable; if it ever fired it would indicate a memory-
  /// safety bug somewhere in this module.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    match &mut self.inner {
      SpillMutInner::Heap(arc) => {
        Arc::get_mut(arc).expect("SpillBytesMut: heap Arc must be unique (refcount 1)")
      }
      #[cfg(any(unix, windows))]
      SpillMutInner::Mmap { map, .. } => {
        let bytes: &mut [u8] = &mut map[..];
        if bytes.is_empty() {
          return &mut [];
        }
        bytemuck::cast_slice_mut(bytes)
      }
    }
  }

  /// Returns `true` if this buffer is backed by an mmap'd tempfile.
  /// `false` if it is heap-backed. Always `false` on
  /// `cfg(not(any(unix, windows)))` (mmap path is compiled out).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_mmapped(&self) -> bool {
    #[cfg(any(unix, windows))]
    {
      matches!(self.inner, SpillMutInner::Mmap { .. })
    }
    #[cfg(not(any(unix, windows)))]
    {
      false
    }
  }

  /// Convert to a [`SpillBytes`] for cheap-clone fan-out.
  ///
  /// Zero-copy on both backends:
  /// - Heap: the underlying `Arc<[T]>` is moved out; refcount is
  ///   still 1 after the move, ready to be cloned by consumers.
  /// - Mmap: the `MmapMut + std::fs::File` pair is wrapped in a
  ///   single `Arc<MmapHandle>`. No data is read or copied.
  pub fn freeze(self) -> SpillBytes<T> {
    let data = match self.inner {
      SpillMutInner::Heap(arc) => SpillBytesData::Heap(arc),
      #[cfg(any(unix, windows))]
      SpillMutInner::Mmap { map, _file } => {
        SpillBytesData::Mmap(Arc::new(MmapHandle { map, _file }))
      }
    };
    SpillBytes {
      data,
      len: self.len,
      _phantom: PhantomData,
    }
  }
}

// SAFETY: a `SpillBytesMut<T>` owns its backing storage uniquely
// (refcount-1 `Arc<[T]>` or per-instance `MmapMut + std::fs::File`).
// Sending the owned handle across threads is safe; both `Arc<[T]>`
// (with `T: Send`) and `MmapMut` are `Send`. We do NOT impl `Sync`:
// `as_mut_slice` exposes `&mut [T]`, whose aliasing semantics
// require unique access.
unsafe impl<T: Pod + Send> Send for SpillBytesMut<T> {}

// ── SpillBytes: read-phase, cheap-clone, Send + Sync ──────────────

/// Frozen, read-only counterpart to [`SpillBytesMut`]. `Clone` is
/// `Arc::clone` on both backends — O(1), no data copy. `Send + Sync`
/// so multiple threads can share the same buffer concurrently.
///
/// Construct via [`SpillBytesMut::freeze`].
pub struct SpillBytes<T> {
  data: SpillBytesData<T>,
  len: usize,
  _phantom: PhantomData<T>,
}

enum SpillBytesData<T> {
  /// `Arc<[T]>` heap backend, produced by [`SpillBytesMut::freeze`] on
  /// the heap path: `SpillBytesMut::zeros` allocates the `Arc<[T]>`
  /// up front (refcount 1) and `freeze` moves it out unchanged.
  Heap(Arc<[T]>),
  /// `Arc<Vec<T>>` heap backend that ADOPTS a caller-owned `Vec<T>`
  /// with no element copy. Produced by [`SpillBytes::from_vec`]: the
  /// public carrier boundary receives an owned `Vec` built from raw
  /// model output, and `Arc::new(vec)` wraps it in place — only the
  /// `Vec`'s three-word header moves into the `Arc` allocation, the
  /// element buffer is untouched, so `as_slice().as_ptr()` equals the
  /// original `Vec::as_ptr()`. `Arc<[T]>::from(vec)` would instead
  /// allocate a fresh `Arc<[T]>` and copy every element (it can't
  /// reuse the `Vec`'s allocation across the `Vec -> [T]` unsizing),
  /// re-materializing the buffer we are trying to hand off cheaply.
  Owned(Arc<Vec<T>>),
  /// Compiled out on `cfg(not(any(unix, windows)))`; see
  /// [`SpillError::UnsupportedTarget`].
  #[cfg(any(unix, windows))]
  Mmap(Arc<MmapHandle>),
}

impl<T> Clone for SpillBytesData<T> {
  fn clone(&self) -> Self {
    match self {
      SpillBytesData::Heap(arc) => SpillBytesData::Heap(Arc::clone(arc)),
      SpillBytesData::Owned(arc) => SpillBytesData::Owned(Arc::clone(arc)),
      #[cfg(any(unix, windows))]
      SpillBytesData::Mmap(arc) => SpillBytesData::Mmap(Arc::clone(arc)),
    }
  }
}

impl<T> Clone for SpillBytes<T> {
  /// O(1): bumps the inner `Arc` refcount. The underlying buffer is
  /// shared with the source.
  fn clone(&self) -> Self {
    Self {
      data: self.data.clone(),
      len: self.len,
      _phantom: PhantomData,
    }
  }
}

impl<T: Pod> SpillBytes<T> {
  /// Wrap an owned heap `Vec<T>` as a heap-backed [`SpillBytes`],
  /// ADOPTING the `Vec`'s existing allocation with **no element copy**.
  ///
  /// `Arc::new(v)` moves only the `Vec`'s three-word header (pointer /
  /// length / capacity) into a fresh `Arc<Vec<T>>` allocation; the
  /// element buffer the `Vec` points at is left exactly where it was,
  /// so [`as_slice`](Self::as_slice)`().as_ptr()` equals the original
  /// `v.as_ptr()`. The result is always the `Owned` heap backend
  /// ([`is_mmapped`](Self::is_mmapped) is `false`).
  ///
  /// This is the genuinely-zero-copy bridge for a call site that
  /// already owns a heap `Vec` and needs the spill-capable carrier
  /// type — no round-trip through [`SpillBytesMut::zeros`] +
  /// `copy_from_slice`, and no `Arc::<[T]>::from(v)` (which would
  /// allocate a fresh `Arc<[T]>` and copy every element, because the
  /// `Vec -> [T]` unsizing cannot reuse the `Vec`'s allocation — and
  /// whose allocation failure aborts the process globally instead of
  /// being recoverable).
  ///
  /// Used at the public `RangeEmbeddings::new` boundary, where the
  /// caller supplies owned `Vec`s built from raw model output: that
  /// path is already heap-resident, so nothing is gained by spilling
  /// it to mmap, and adopting the buffer avoids the transient heap
  /// duplication a copy would cause. The internal `build_range` path
  /// instead `freeze`s its existing `SpillBytesMut`, so long ranges
  /// retain their mmap backing all the way through the carrier.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn from_vec(v: Vec<T>) -> Self {
    let len = v.len();
    Self {
      data: SpillBytesData::Owned(Arc::new(v)),
      len,
      _phantom: PhantomData,
    }
  }

  /// Number of `T` cells in the buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn len(&self) -> usize {
    self.len
  }

  /// `true` if the buffer is empty.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// Borrow the buffer as `&[T]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn as_slice(&self) -> &[T] {
    match &self.data {
      SpillBytesData::Heap(arc) => arc,
      SpillBytesData::Owned(vec) => &vec[..],
      #[cfg(any(unix, windows))]
      SpillBytesData::Mmap(handle) => {
        let bytes: &[u8] = &handle.map[..];
        if bytes.is_empty() {
          return &[];
        }
        bytemuck::cast_slice(bytes)
      }
    }
  }

  /// Returns `true` if this buffer is backed by an mmap'd tempfile.
  /// `false` if it is heap-backed (either the `Heap` `Arc<[T]>` or the
  /// `Owned` `Arc<Vec<T>>` backend). Always `false` on
  /// `cfg(not(any(unix, windows)))` (mmap path is compiled out).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_mmapped(&self) -> bool {
    #[cfg(any(unix, windows))]
    {
      matches!(self.data, SpillBytesData::Mmap(_))
    }
    #[cfg(not(any(unix, windows)))]
    {
      false
    }
  }
}

// SAFETY: `SpillBytes<T>` only exposes `&[T]` (no mutation reaches
// the buffer once frozen). The `Heap` variant wraps `Arc<[T]>` and the
// `Owned` variant wraps `Arc<Vec<T>>`; both are `Send + Sync` for
// `T: Send + Sync` by the standard library's own impls (no raw
// pointers, no interior mutability). The mmap variant wraps
// `Arc<MmapHandle>`, which contains `MmapMut + std::fs::File`; both
// are `Send + Sync` for read-only access (`memmapix` exposes the
// same `Send + Sync` semantics as `memmap2`). For `T: Pod` (= plain
// bytes, no interior pointers), `T: Send + Sync` always holds.
unsafe impl<T: Pod + Send + Sync> Send for SpillBytes<T> {}
unsafe impl<T: Pod + Send + Sync> Sync for SpillBytes<T> {}

/// `Deref` so `SpillBytes<T>` substitutes for `Arc<[T]>` /
/// `&[T]` at most call sites: indexing (`buf[i]`), slicing
/// (`&buf[..]`), `.iter()`, `.len()` (also defined directly on
/// `SpillBytes`; the inherent method takes priority but the
/// deref'd slice version is equivalent), and so on. Equivalent
/// to `as_slice()` but ergonomic.
impl<T: Pod> core::ops::Deref for SpillBytes<T> {
  type Target = [T];
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn deref(&self) -> &[T] {
    self.as_slice()
  }
}

impl<T: Pod + core::fmt::Debug> core::fmt::Debug for SpillBytes<T> {
  /// Length-tagged backend summary plus a bounded head (first 8
  /// cells). Avoids formatting an mmap-backed multi-GB buffer in
  /// full — `as_slice()`'s `Debug` would walk every element — while
  /// keeping the small-grid test-debug output useful.
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    const HEAD: usize = 8;
    let s = self.as_slice();
    let head_n = s.len().min(HEAD);
    f.debug_struct("SpillBytes")
      .field("len", &self.len)
      .field("backend", &if self.is_mmapped() { "mmap" } else { "heap" })
      .field("head", &&s[..head_n])
      .finish()
  }
}

impl<T: Pod + core::fmt::Debug> core::fmt::Debug for SpillBytesMut<T> {
  /// Same length-tagged summary as `SpillBytes`; full contents
  /// elided for the same reason.
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("SpillBytesMut")
      .field("len", &self.len)
      .field("backend", &if self.is_mmapped() { "mmap" } else { "heap" })
      .finish()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// `SpillOptions::new()` is `const fn` and produces the documented
  /// default values.
  #[test]
  fn default_options_const_fn() {
    const OPTS: SpillOptions = SpillOptions::new();
    assert_eq!(
      OPTS.threshold_bytes(),
      SpillOptions::DEFAULT_THRESHOLD_BYTES
    );
    assert_eq!(SpillOptions::DEFAULT_THRESHOLD_BYTES, 64 * 1024 * 1024);
  }

  /// `with_threshold_bytes` is `const fn`; constructing a tuned
  /// `SpillOptions` at compile time is supported.
  #[test]
  fn const_fn_builder() {
    const OPTS: SpillOptions = SpillOptions::new().with_threshold_bytes(1024);
    assert_eq!(OPTS.threshold_bytes(), 1024);
    assert!(OPTS.spill_dir().is_none());
  }

  #[test]
  fn set_threshold_bytes_chains() {
    let mut opts = SpillOptions::new();
    opts
      .set_threshold_bytes(42)
      .set_spill_dir(Some("/tmp/dia".into()));
    assert_eq!(opts.threshold_bytes(), 42);
    assert_eq!(opts.spill_dir(), Some(Path::new("/tmp/dia")));
  }

  /// `SpillBytesMut::zeros(0, _)` returns an empty heap buffer,
  /// never touching mmap (mmap of length 0 is `EINVAL` on most
  /// platforms).
  #[test]
  fn zeros_zero_returns_heap_empty() {
    let opts = SpillOptions::default();
    let v: SpillBytesMut<f64> = SpillBytesMut::zeros(0, &opts).expect("zero-length must succeed");
    assert_eq!(v.len(), 0);
    assert!(v.is_empty());
    assert_eq!(v.as_slice().len(), 0);
    assert!(!v.is_mmapped());
  }

  /// Below threshold: heap-backed.
  #[test]
  fn small_allocation_uses_heap() {
    // Default threshold is 64 MiB; a 1 KiB f64 buffer is well under.
    let opts = SpillOptions::default();
    let v: SpillBytesMut<f64> = SpillBytesMut::zeros(128, &opts).expect("alloc");
    assert_eq!(v.len(), 128);
    assert!(!v.is_mmapped());
    assert!(v.as_slice().iter().all(|&x| x == 0.0));
  }

  /// Reads and writes round-trip through both backends. The two
  /// allocations use different `SpillOptions` instances — no shared
  /// state means no cross-test contamination.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "miri does not support fcntl(F_PREALLOCATE) / mmap; mmap-path tests cannot run"
  )]
  fn read_write_roundtrip_both_backends() {
    let mmap_opts = SpillOptions::default().with_threshold_bytes(0);
    let mut v: SpillBytesMut<f64> = SpillBytesMut::zeros(64, &mmap_opts).expect("mmap alloc");
    assert!(v.is_mmapped(), "should be mmap-backed at threshold=0");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = i as f64 * 1.5;
    }
    for (i, &x) in v.as_slice().iter().enumerate() {
      assert_eq!(x, i as f64 * 1.5);
    }
    drop(v);

    let heap_opts = SpillOptions::default().with_threshold_bytes(usize::MAX);
    let mut v: SpillBytesMut<f64> = SpillBytesMut::zeros(64, &heap_opts).expect("heap alloc");
    assert!(
      !v.is_mmapped(),
      "should be heap-backed at threshold=usize::MAX"
    );
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = i as f64 * 1.5;
    }
    for (i, &x) in v.as_slice().iter().enumerate() {
      assert_eq!(x, i as f64 * 1.5);
    }
  }

  /// Differential test: heap and mmap backends must produce
  /// bit-identical contents for the same write sequence.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "miri does not support fcntl(F_PREALLOCATE) / mmap; mmap-path tests cannot run"
  )]
  fn heap_mmap_differential_bit_equal() {
    fn fill_and_collect<F: FnOnce(&mut SpillBytesMut<f64>)>(threshold: usize, fill: F) -> Vec<f64> {
      let opts = SpillOptions::new().with_threshold_bytes(threshold);
      let mut v: SpillBytesMut<f64> = SpillBytesMut::zeros(1024, &opts).expect("alloc");
      fill(&mut v);
      v.as_slice().to_vec()
    }
    let fill_pattern = |v: &mut SpillBytesMut<f64>| {
      for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
        *slot = (i as f64).sqrt() + 0.001 * (i as f64);
      }
    };
    let heap = fill_and_collect(usize::MAX, fill_pattern);
    let mmap = fill_and_collect(0, fill_pattern);
    assert_eq!(
      heap.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
      mmap.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
      "heap and mmap backends must produce bit-equal contents"
    );
  }

  /// Size-overflow surfaces a typed error instead of panicking.
  #[test]
  fn size_overflow_returns_typed_error() {
    let opts = SpillOptions::default();
    let r: Result<SpillBytesMut<f64>, _> = SpillBytesMut::zeros(usize::MAX / 4, &opts);
    let err = r.unwrap_err();
    assert!(
      matches!(err, SpillError::SizeOverflow { .. }),
      "got {err:?}"
    );
  }

  /// `Vec<u8>`-as-mask works for the boolean-mask migration. `bool`
  /// is not `Pod` so the masks switch to `u8` (0/1).
  #[test]
  fn u8_mask_roundtrip() {
    let opts = SpillOptions::default();
    let mut v: SpillBytesMut<u8> = SpillBytesMut::zeros(16, &opts).expect("alloc");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = if i.is_multiple_of(2) { 1 } else { 0 };
    }
    let s = v.as_slice();
    for i in 0..16 {
      assert_eq!(s[i], if i.is_multiple_of(2) { 1 } else { 0 });
    }
  }

  /// `f32` cells (the reconstruct grid is f32). Confirm the type works.
  #[test]
  fn f32_roundtrip() {
    let opts = SpillOptions::default();
    let mut v: SpillBytesMut<f32> = SpillBytesMut::zeros(8, &opts).expect("alloc");
    let target: [f32; 8] = [
      0.0,
      1.0,
      0.5,
      -0.25,
      1e10,
      -1e10,
      f32::EPSILON,
      -f32::EPSILON,
    ];
    v.as_mut_slice().copy_from_slice(&target);
    assert_eq!(v.as_slice(), &target);
  }

  /// Distinct `SpillOptions` values produce distinct backend
  /// choices on the same allocation size.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "miri does not support fcntl(F_PREALLOCATE) / mmap; mmap-path tests cannot run"
  )]
  fn distinct_options_pick_distinct_backends() {
    let mmap_opts = SpillOptions::new().with_threshold_bytes(0);
    let v: SpillBytesMut<f64> = SpillBytesMut::zeros(64, &mmap_opts).expect("mmap alloc");
    assert!(v.is_mmapped());
    drop(v);

    let heap_opts = SpillOptions::new().with_threshold_bytes(usize::MAX);
    let v: SpillBytesMut<f64> = SpillBytesMut::zeros(64, &heap_opts).expect("heap alloc");
    assert!(!v.is_mmapped());
  }

  // ── SpillBytes: freeze + cheap-clone fan-out ────────────────────

  /// Freeze on the heap path preserves contents and the `Heap`
  /// backend tag; subsequent clones are cheap (Arc-shared).
  #[test]
  fn freeze_heap_preserves_data_and_backend() {
    let opts = SpillOptions::default().with_threshold_bytes(usize::MAX);
    let mut v: SpillBytesMut<f64> = SpillBytesMut::zeros(32, &opts).expect("alloc");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = i as f64;
    }
    assert!(!v.is_mmapped());
    let frozen = v.freeze();
    assert!(!frozen.is_mmapped());
    assert_eq!(frozen.len(), 32);
    let expected: Vec<f64> = (0..32).map(|i| i as f64).collect();
    assert_eq!(frozen.as_slice(), expected.as_slice());
  }

  /// Freeze on the mmap path preserves contents and the `Mmap`
  /// backend tag.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "miri does not support fcntl(F_PREALLOCATE) / mmap; mmap-path tests cannot run"
  )]
  fn freeze_mmap_preserves_data_and_backend() {
    let opts = SpillOptions::default().with_threshold_bytes(0);
    let mut v: SpillBytesMut<f64> = SpillBytesMut::zeros(32, &opts).expect("alloc");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = i as f64 * 0.5;
    }
    assert!(v.is_mmapped());
    let frozen = v.freeze();
    assert!(frozen.is_mmapped());
    assert_eq!(frozen.len(), 32);
    let expected: Vec<f64> = (0..32).map(|i| i as f64 * 0.5).collect();
    assert_eq!(frozen.as_slice(), expected.as_slice());
  }

  /// Cloning a frozen buffer shares storage: every clone observes
  /// the same data, and the `as_slice` pointers are equal (the
  /// classic Arc-share assertion).
  #[test]
  fn clone_shares_heap_storage() {
    let opts = SpillOptions::default().with_threshold_bytes(usize::MAX);
    let mut v: SpillBytesMut<f64> = SpillBytesMut::zeros(16, &opts).expect("alloc");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = (i as f64).sqrt();
    }
    let original = v.freeze();
    let a = original.clone();
    let b = original.clone();
    assert_eq!(a.as_slice(), b.as_slice());
    // Same underlying allocation: identical pointer.
    assert!(std::ptr::eq(a.as_slice().as_ptr(), b.as_slice().as_ptr()));
    assert!(std::ptr::eq(
      a.as_slice().as_ptr(),
      original.as_slice().as_ptr()
    ));
  }

  /// Same shared-storage assertion for the mmap backend.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "miri does not support fcntl(F_PREALLOCATE) / mmap; mmap-path tests cannot run"
  )]
  fn clone_shares_mmap_storage() {
    let opts = SpillOptions::default().with_threshold_bytes(0);
    let mut v: SpillBytesMut<f64> = SpillBytesMut::zeros(16, &opts).expect("alloc");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = i as f64;
    }
    let original = v.freeze();
    let a = original.clone();
    let b = original.clone();
    assert_eq!(a.as_slice(), b.as_slice());
    assert!(std::ptr::eq(a.as_slice().as_ptr(), b.as_slice().as_ptr()));
    assert!(std::ptr::eq(
      a.as_slice().as_ptr(),
      original.as_slice().as_ptr()
    ));
  }

  /// Clones of a `SpillBytes` keep the buffer alive after the
  /// original is dropped — `Arc` refcounting works as expected.
  #[test]
  fn clone_outlives_original() {
    let opts = SpillOptions::default();
    let mut v: SpillBytesMut<f64> = SpillBytesMut::zeros(8, &opts).expect("alloc");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = (i as f64) * 2.0;
    }
    let original = v.freeze();
    let clone = original.clone();
    drop(original);
    let expected: Vec<f64> = (0..8).map(|i| (i as f64) * 2.0).collect();
    assert_eq!(clone.as_slice(), expected.as_slice());
  }

  /// `SpillBytes` is `Send + Sync` so the same frozen buffer can
  /// be shared across threads without further synchronization.
  #[test]
  fn send_sync_fan_out_across_threads() {
    let opts = SpillOptions::default();
    let mut v: SpillBytesMut<f64> = SpillBytesMut::zeros(64, &opts).expect("alloc");
    for (i, slot) in v.as_mut_slice().iter_mut().enumerate() {
      *slot = i as f64;
    }
    let frozen = v.freeze();
    let mut handles = Vec::new();
    for _ in 0..4 {
      let c = frozen.clone();
      handles.push(std::thread::spawn(move || {
        let s = c.as_slice();
        let mut sum = 0.0;
        for &x in s {
          sum += x;
        }
        sum
      }));
    }
    let want = (0..64).map(|i| i as f64).sum::<f64>();
    for h in handles {
      assert_eq!(h.join().unwrap(), want);
    }
  }

  /// Compile-time check: `SpillBytes<f64>` must be `Send + Sync`.
  /// The `static_assert`-style pattern uses a generic helper that
  /// only compiles when the bound holds.
  #[test]
  fn spill_bytes_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SpillBytes<f64>>();
    assert_send_sync::<SpillBytes<f32>>();
    assert_send_sync::<SpillBytes<u8>>();
  }

  // ── SpillBytes::from_vec: zero-copy Vec adoption ────────────────

  /// FINDING 2: `from_vec` ADOPTS the `Vec`'s existing allocation with
  /// no element copy — the slice pointer it exposes equals the
  /// original `Vec::as_ptr()`. Previously `Arc::<[T]>::from(v)`
  /// allocated a fresh `Arc<[T]>` and copied every element (and an
  /// allocation failure there aborts the process). The pointer-identity
  /// assertion is the load-bearing proof that the public carrier
  /// boundary no longer transiently duplicates the f64 segmentations /
  /// f32 embeddings.
  #[test]
  fn from_vec_is_zero_copy_pointer_identity() {
    let v: Vec<f64> = (0..1024).map(|i| i as f64 * 0.25).collect();
    let orig_ptr = v.as_ptr();
    let orig_len = v.len();
    let sb = SpillBytes::from_vec(v);
    assert_eq!(sb.len(), orig_len);
    assert!(!sb.is_mmapped(), "from_vec is always heap-backed");
    // The slice the carrier exposes points at the SAME allocation the
    // original Vec owned — no element copy occurred.
    assert!(
      std::ptr::eq(sb.as_slice().as_ptr(), orig_ptr),
      "from_vec must adopt the Vec's buffer (zero-copy), not copy it"
    );
    // Contents intact.
    for (i, &x) in sb.as_slice().iter().enumerate() {
      assert_eq!(x, i as f64 * 0.25);
    }
  }

  /// Same pointer-identity guarantee for the f32 embeddings carrier
  /// (the other buffer `RangeEmbeddings::new` wraps via `from_vec`).
  #[test]
  fn from_vec_zero_copy_f32() {
    let v: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let orig_ptr = v.as_ptr();
    let sb = SpillBytes::from_vec(v);
    assert!(std::ptr::eq(sb.as_slice().as_ptr(), orig_ptr));
  }

  /// A `from_vec` carrier clones cheaply (Arc refcount bump) and every
  /// clone shares the adopted buffer — same pointer, and the buffer
  /// outlives the original. Proves the `Owned(Arc<Vec<T>>)` backend
  /// keeps `SpillBytes`'s O(1)-clone fan-out contract.
  #[test]
  fn from_vec_clone_shares_adopted_buffer() {
    let v: Vec<u8> = (0..64).map(|i| i as u8).collect();
    let orig_ptr = v.as_ptr();
    let original = SpillBytes::from_vec(v);
    let a = original.clone();
    let b = original.clone();
    assert!(std::ptr::eq(a.as_slice().as_ptr(), orig_ptr));
    assert!(std::ptr::eq(a.as_slice().as_ptr(), b.as_slice().as_ptr()));
    drop(original);
    // Clones keep the buffer alive after the original is dropped.
    assert_eq!(a.as_slice(), b.as_slice());
    for (i, &x) in a.as_slice().iter().enumerate() {
      assert_eq!(x, i as u8);
    }
  }

  /// An empty `Vec` adopts cleanly through `from_vec` (length 0, heap
  /// backend, no panic). Pointer identity is not asserted for the empty
  /// case — an empty `Vec`'s dangling `as_ptr()` is not a meaningful
  /// allocation to compare.
  #[test]
  fn from_vec_empty() {
    let v: Vec<f64> = Vec::new();
    let sb = SpillBytes::from_vec(v);
    assert_eq!(sb.len(), 0);
    assert!(sb.is_empty());
    assert!(!sb.is_mmapped());
    assert_eq!(sb.as_slice(), &[] as &[f64]);
  }
}
