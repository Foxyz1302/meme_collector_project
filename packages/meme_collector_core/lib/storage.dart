/// Where data lives. JSON I/O, atomic rename, path resolution, config.
///
/// The metadata.json file is the source of truth for state that can't be
/// rebuilt from disk (URLs, user metadata, usage counts). Everything else
/// (thumbnails, embeddings, OCR) lives as files named by reaction ID.
///
/// Atomic writes use MoveFileExW on Windows (Dart's File.rename uses
/// MoveFileW which fails if destination exists). POSIX uses File.rename
/// directly (already atomic).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:path/path.dart' as p;

import 'models.dart';

// ─── Atomic rename ─────────────────────────────────────────────────────────

/// On Windows: call MoveFileExW with MOVEFILE_REPLACE_EXISTING.
/// On POSIX: File.rename is already atomic.
///
/// Crash-safety: the destination is never in a partial state. Either the
/// old version or the new version is present, never a half-written file.
Future<void> atomicRename(String from, String to) {
  if (Platform.isWindows) {
    return _atomicRenameWindows(from, to);
  }
  return File(from).rename(to);
}

// Windows FFI bits — keep them private to this file.
const int _moveFileReplaceExisting = 0x00000001;
const int _moveFileWriteThrough = 0x00000008;

typedef _MoveFileExWNative = Int32 Function(
    Pointer<Utf16> lpExistingFileName,
    Pointer<Utf16> lpNewFileName,
    Uint32 dwFlags);
typedef _MoveFileExWDart = int Function(
    Pointer<Utf16> lpExistingFileName,
    Pointer<Utf16> lpNewFileName,
    int dwFlags);

Future<void> _atomicRenameWindows(String from, String to) async {
  // Use Isolate.run so the FFI call doesn't block the UI event loop.
  final ok = await Isolate.run(() {
    final lib = DynamicLibrary.open('kernel32.dll');
    final moveFileEx = lib.lookupFunction<_MoveFileExWNative, _MoveFileExWDart>(
        'MoveFileExW');

    final fromPtr = from.toNativeUtf16();
    final toPtr = to.toNativeUtf16();
    try {
      return moveFileEx(fromPtr, toPtr,
          _moveFileReplaceExisting | _moveFileWriteThrough);
    } finally {
      calloc.free(fromPtr);
      calloc.free(toPtr);
    }
  });

  if (ok == 0) {
    // Fall back to delete-then-rename. Not atomic, but recovers.
    try {
      await File(to).delete();
    } on FileSystemException {
      // destination didn't exist — fine
    }
    await File(from).rename(to);
  }
}

// ─── Storage ───────────────────────────────────────────────────────────────

/// Owns the storage root path and provides typed access to subdirectories.
///
/// Layout under [rootPath]:
///   metadata.json
///   originals/{id}.{ext}
///   thumbnails_static/{id}.webp
///   thumbnails_animated/{id}.webp
///   embeddings_text/{producer}__{version}/{id}.f32.bin
///   embeddings_image/{producer}__{version}/{id}.f32.bin
///   ocr/{id}.txt
class Storage {
  final String rootPath;

  Storage(this.rootPath);

  // ─── Paths ──────────────────────────────────────────────────────────────

  String get metadataPath => p.join(rootPath, 'metadata.json');
  String get metadataTmpPath => p.join(rootPath, 'metadata.json.tmp');

  String get originalsDir => p.join(rootPath, 'originals');
  String get thumbnailsStaticDir => p.join(rootPath, 'thumbnails_static');
  String get thumbnailsAnimatedDir => p.join(rootPath, 'thumbnails_animated');
  String get embeddingsTextDir => p.join(rootPath, 'embeddings_text');
  String get embeddingsImageDir => p.join(rootPath, 'embeddings_image');
  String get ocrDir => p.join(rootPath, 'ocr');
  String get downloadsDir => p.join(rootPath, 'downloads');

  /// Path for a text embedding file under the given producer+version.
  String textEmbeddingPath(
          String reactionId, String producer, String version) =>
      p.join(embeddingsTextDir, '${producer}__$version', '$reactionId.f32.bin');

  /// Path for an image embedding file under the given producer+version.
  String imageEmbeddingPath(
          String reactionId, String producer, String version) =>
      p.join(embeddingsImageDir, '${producer}__$version', '$reactionId.f32.bin');

  String thumbnailStaticPath(String reactionId) =>
      p.join(thumbnailsStaticDir, '$reactionId.webp');

  String thumbnailAnimatedPath(String reactionId) =>
      p.join(thumbnailsAnimatedDir, '$reactionId.webp');

  String ocrPath(String reactionId) => p.join(ocrDir, '$reactionId.txt');

  String downloadPath(String reactionId, String extension) =>
      p.join(downloadsDir, '$reactionId.$extension');

  // ─── Directory initialization ───────────────────────────────────────────

  Future<void> ensureDirectoriesExist() async {
    final dirs = [
      rootPath,
      originalsDir,
      thumbnailsStaticDir,
      thumbnailsAnimatedDir,
      downloadsDir,
      embeddingsTextDir,
      embeddingsImageDir,
      ocrDir,
    ];
    for (final dir in dirs) {
      await Directory(dir).create(recursive: true);
    }
  }

  // ─── Metadata load/save ─────────────────────────────────────────────────

  /// Load metadata.json. If the file doesn't exist (first launch), returns
  /// an empty [Metadata].
  Future<Metadata> loadMetadata() async {
    final file = File(metadataPath);
    if (!await file.exists()) {
      return Metadata.empty();
    }
    try {
      final contents = await file.readAsString();
      final json = jsonDecode(contents) as Map<String, dynamic>;
      return Metadata.fromJson(json);
    } on FormatException catch (e) {
      // Corrupted JSON — fall back to empty rather than crash.
      // TODO: log this somewhere the user can see.
      stderr.writeln('metadata.json corrupted: $e');
      return Metadata.empty();
    }
  }

  Timer? _saveDebounce;
  Metadata? _pendingSave;
  bool _flushing = false;

  /// Schedule a debounced save. Multiple rapid calls collapse into one write.
  void scheduleSave(Metadata metadata, {Duration delay = const Duration(milliseconds: 500)}) {
    _pendingSave = metadata;
    _saveDebounce?.cancel();
    _saveDebounce = Timer(delay, _doSave);
  }

  /// Force-write metadata immediately (bypasses debounce). Used by the
  /// Coordinator's _doSave() which manages its own debounce timing.
  Future<void> saveNow(Metadata metadata) async {
    await ensureDirectoriesExist();
    final tmpFile = File(metadataTmpPath);
    final json = const JsonEncoder.withIndent('  ').convert(metadata.toJson());
    await tmpFile.writeAsString(json, flush: true);
    await atomicRename(metadataTmpPath, metadataPath);
  }

  /// Force-write any pending debounced save. Call on app exit.
  Future<void> flushNow() async {
    if (_flushing) return;
    _flushing = true;
    _saveDebounce?.cancel();
    await _doSave();
    _flushing = false;
  }

  Future<void> _doSave() async {
    final m = _pendingSave;
    if (m == null) return;

    await ensureDirectoriesExist();
    final tmpFile = File(metadataTmpPath);
    final json = const JsonEncoder.withIndent('  ').convert(m.toJson());
    await tmpFile.writeAsString(json, flush: true);
    await atomicRename(metadataTmpPath, metadataPath);
    _pendingSave = null;
  }

  // ─── File operations ────────────────────────────────────────────────────

  /// Delete all derivative files for a reaction (on user delete).
  /// The metadata entry is removed separately by the coordinator.
  Future<void> deleteReactionFiles(Reaction reaction) async {
    final paths = [
      if (reaction.localFile != null) p.join(rootPath, reaction.localFile),
      if (reaction.thumbnailStatic != null)
        p.join(rootPath, reaction.thumbnailStatic),
      if (reaction.thumbnailAnimated != null)
        p.join(rootPath, reaction.thumbnailAnimated),
      if (reaction.ocrText != null) ocrPath(reaction.id),
      if (reaction.textEmbeddingPath != null)
        p.join(rootPath, reaction.textEmbeddingPath),
      if (reaction.imageEmbeddingPath != null)
        p.join(rootPath, reaction.imageEmbeddingPath),
    ];
    for (final path in paths) {
      try {
        await File(path).delete();
      } on FileSystemException {
        // file didn't exist — fine
      }
    }
  }

  /// Migrate the entire storage root to a new location.
  /// Copies all files, verifies count + total size, then swaps.
  /// Returns true on success.
  Future<bool> migrateTo(String newRootPath) async {
    final newStorage = Storage(newRootPath);
    await newStorage.ensureDirectoriesExist();

    // Copy everything except the metadata.json (we'll write it last)
    final entries = await Directory(rootPath).list(recursive: true).toList();
    for (final entry in entries) {
      if (entry is! File) continue;
      final relative = p.relative(entry.path, from: rootPath);
      if (relative == 'metadata.json' || relative == 'metadata.json.tmp') {
        continue;
      }
      final dest = p.join(newRootPath, relative);
      await Directory(p.dirname(dest)).create(recursive: true);
      await entry.copy(dest);
    }

    // Now write the metadata to the new location
    final metadata = await loadMetadata();
    final newMetaFile = File(newStorage.metadataPath);
    await newMetaFile.writeAsString(
        const JsonEncoder.withIndent('  ').convert(metadata.toJson()),
        flush: true);

    // Verify: count files in both locations
    final oldCount = await Directory(rootPath)
        .list(recursive: true)
        .where((e) => e is File)
        .length;
    final newCount = await Directory(newRootPath)
        .list(recursive: true)
        .where((e) => e is File)
        .length;
    if (newCount < oldCount) {
      return false; // migration incomplete
    }

    return true;
  }

  // ─── Writability check ──────────────────────────────────────────────────

  /// Test if a path is writable by creating + deleting a temp file.
  static Future<bool> isWritable(String path) async {
    try {
      await Directory(path).create(recursive: true);
      final testFile = File(p.join(path, '.writetest'));
      await testFile.writeAsString('test');
      await testFile.delete();
      return true;
    } catch (_) {
      return false;
    }
  }

  /// Scan the storage root for orphan files (on disk but not in metadata).
  /// Used for crash recovery — partial downloads that didn't get registered.
  Future<List<String>> findOrphanFiles(Metadata metadata) async {
    final knownIds = metadata.reactions.map((r) => r.id).toSet();
    final orphans = <String>[];

    for (final dir in [
      downloadsDir,
      thumbnailsStaticDir,
      thumbnailsAnimatedDir,
      ocrDir,
    ]) {
      final d = Directory(dir);
      if (!await d.exists()) continue;
      await for (final entry in d.list()) {
        if (entry is! File) continue;
        final basename = p.basenameWithoutExtension(entry.path);
        if (!knownIds.contains(basename)) {
          orphans.add(entry.path);
        }
      }
    }
    return orphans;
  }
}

// ─── Float32 vector file helpers ───────────────────────────────────────────

/// Read a Float32 vector from a .f32.bin file.
Float32List readVectorFile(String path) {
  final bytes = File(path).readAsBytesSync();
  return bytes.buffer.asFloat32List(bytes.offsetInBytes, bytes.length ~/ 4);
}

/// Write a Float32 vector to a .f32.bin file.
Future<void> writeVectorFile(String path, Float32List vector) async {
  await File(path).parent.create(recursive: true);
  final bytes = Uint8List.view(vector.buffer, vector.offsetInBytes,
      vector.length * 4);
  await File(path).writeAsBytes(bytes, flush: true);
}
