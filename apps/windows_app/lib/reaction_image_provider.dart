/// Custom ImageProvider for reactions.
///
/// Priority:
///   1. thumbnailAnimated (if animated previews enabled + exists)
///   2. thumbnailStatic (if exists)
///   3. localFile (if downloaded)
///   4. Network URL via dio (instant display, with download progress)
///
/// Keyed on (reactionId, thumbnailPath) so Flutter's ImageCache auto-evicts
/// when the thumbnail becomes available → smooth swap from network to local.
///
/// Captures dimensions during decode and reports them via onDimensions callback
/// so the masonry layout can adjust before ingest runs.

library;

import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/painting.dart';

import 'package:meme_collector_core/meme_collector_core.dart';

@immutable
class ReactionImageProvider extends ImageProvider<ReactionImageProvider> {
  final String reactionId;
  final String url;
  final String? thumbnailStaticPath;
  final String? thumbnailAnimatedPath;
  final String? localFilePath;
  final bool animatedPreviewsEnabled;
  final double scale;

  /// Called when the image decodes with its natural dimensions.
  /// Used to capture width/height for masonry layout before ingest runs.
  final void Function(int width, int height)? onDimensions;

  const ReactionImageProvider({
    required this.reactionId,
    required this.url,
    this.thumbnailStaticPath,
    this.thumbnailAnimatedPath,
    this.localFilePath,
    this.animatedPreviewsEnabled = false,
    this.scale = 1.0,
    this.onDimensions,
  });

  /// The key that determines cache identity.
  /// When thumbnailPath changes (null → path), this changes → cache evicts → reload.
  String get _cacheKey {
    if (animatedPreviewsEnabled && thumbnailAnimatedPath != null) {
      return 'file:$thumbnailAnimatedPath';
    }
    if (thumbnailStaticPath != null) return 'file:$thumbnailStaticPath';
    if (localFilePath != null) return 'file:$localFilePath';
    return 'url:$url';
  }

  @override
  Future<ReactionImageProvider> obtainKey(ImageConfiguration configuration) {
    return SynchronousFuture(this);
  }

  @override
  ImageStreamCompleter loadImage(
      ReactionImageProvider key, ImageDecoderCallback decode) {
    final chunkEvents = StreamController<ImageChunkEvent>();

    final codec = _loadAsync(key, decode, chunkEvents);

    return MultiFrameImageStreamCompleter(
      codec: codec,
      chunkEvents: chunkEvents.stream,
      scale: key.scale,
      debugLabel: key._cacheKey,
      informationCollector: () => [
        DiagnosticsProperty<String>('Cache key', key._cacheKey),
        DiagnosticsProperty<String>('URL', key.url),
      ],
    );
  }

  Future<ui.Codec> _loadAsync(
    ReactionImageProvider key,
    ImageDecoderCallback decode,
    StreamController<ImageChunkEvent> chunkEvents,
  ) async {
    try {
      // Priority 1: animated thumbnail (if enabled)
      if (key.animatedPreviewsEnabled && key.thumbnailAnimatedPath != null) {
        return await _decodeFile(key.thumbnailAnimatedPath!, decode);
      }

      // Priority 2: static thumbnail
      if (key.thumbnailStaticPath != null) {
        return await _decodeFile(key.thumbnailStaticPath!, decode);
      }

      // Priority 3: local file (downloaded but not yet thumbnailed)
      if (key.localFilePath != null) {
        return await _decodeFile(key.localFilePath!, decode);
      }

      // Priority 4: network URL via dio (with progress events)
      return await _decodeNetwork(key.url, decode, chunkEvents);
    } catch (e) {
      chunkEvents.close();
      rethrow;
    }
  }

  Future<ui.Codec> _decodeFile(
      String path, ImageDecoderCallback decode) async {
    final buffer = await ui.ImmutableBuffer.fromFilePath(path);
    final codec = await decode(buffer);
    _reportDimensions(codec);
    return codec;
  }

  Future<ui.Codec> _decodeNetwork(
    String url,
    ImageDecoderCallback decode,
    StreamController<ImageChunkEvent> chunkEvents,
  ) async {
    // Download to a temp file via dio, then decode
    final tempDir = await Directory.systemTemp.createTemp('reaction_img');
    final tempFile = File('${tempDir.path}/img');

    final dio = Dio();
    await dio.download(
      url,
      tempFile.path,
      onReceiveProgress: (received, total) {
        if (total > 0) {
          chunkEvents.add(ImageChunkEvent(
            cumulativeBytesLoaded: received,
            expectedTotalBytes: total,
          ));
        }
      },
    );

    final buffer = await ui.ImmutableBuffer.fromFilePath(tempFile.path);
    final codec = await decode(buffer);
    _reportDimensions(codec);

    // Clean up temp file (the real download lives in downloads/{id}.{ext}
    // once the ingest pipeline runs)
    try {
      await tempFile.delete();
      await tempDir.delete();
    } catch (_) {}

    return codec;
  }

  void _reportDimensions(ui.Codec codec) {
    if (onDimensions != null) {
      // codec.width/height are available immediately after decode
      onDimensions!(codec.width, codec.height);
    }
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is ReactionImageProvider &&
        other.reactionId == reactionId &&
        other._cacheKey == _cacheKey;
  }

  @override
  int get hashCode => Object.hash(reactionId, _cacheKey);

  @override
  String toString() =>
      'ReactionImageProvider($reactionId, key: $_cacheKey)';
}
