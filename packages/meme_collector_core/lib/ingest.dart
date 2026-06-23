/// URL parsing + ffmpeg wrapper + ingest pipeline state machine.
///
/// Pure Dart (no Flutter deps). The ingest pipeline takes a Reaction from
/// "queued" to "ready" through these stages:
///
///   queued → downloading → downloaded → thumbnailing → embedding → ready
///                                                                      ↘ failed
///
/// Each stage updates the Reaction's status + progress and emits events
/// that the Coordinator forwards to the UI.
///
/// The pipeline is designed to be resumable: if the app crashes mid-ingest,
/// on next launch the Coordinator scans for reactions with status NOT IN
/// (ready, failed) and re-queues them. Stages are idempotent (re-running
/// just overwrites the output file).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:html/parser.dart' as html_parser;
import 'package:path/path.dart' as p;
import 'package:uuid/uuid.dart';

import 'inference.dart';
import 'models.dart';
import 'storage.dart';

// ─── URL Normalizer ─────────────────────────────────────────────────────────

/// Result of URL validation — checks if a URL is supported.
class ValidationResult {
  /// True if the URL can be added as a reaction.
  final bool isValid;

  /// Human-readable reason if invalid.
  final String? reason;

  /// The detected platform (if recognized).
  final SourcePlatform? platform;

  /// The detected media type (if determinable from URL).
  final MediaType? mediaType;

  const ValidationResult({
    required this.isValid,
    this.reason,
    this.platform,
    this.mediaType,
  });

  const ValidationResult.valid({
    required this.platform,
    this.mediaType,
  })  : isValid = true,
        reason = null;

  const ValidationResult.invalid(this.reason)
      : isValid = false,
        platform = null,
        mediaType = null;
}

/// Detected media type from URL or content-type.
enum MediaType {
  image,
  animatedImage,
  video,
  unknown;

  static MediaType? fromExtension(String ext) {
    final e = ext.toLowerCase();
    if ({'.gif'}.contains(e)) return MediaType.animatedImage;
    if ({'.png', '.jpg', '.jpeg', '.webp', '.bmp'}.contains(e)) {
      return MediaType.image;
    }
    if ({'.mp4', '.webm', '.mov', '.mkv', '.avi'}.contains(e)) {
      return MediaType.video;
    }
    return null;
  }

  static MediaType? fromMimeType(String? mime) {
    if (mime == null) return null;
    final m = mime.toLowerCase();
    if (m == 'image/gif') return MediaType.animatedImage;
    if (m.startsWith('image/')) return MediaType.image;
    if (m.startsWith('video/')) return MediaType.video;
    return MediaType.unknown;
  }
}

/// Validates whether a URL can be added as a reaction.
///
/// Quick checks (no network):
///   - Is it a valid URL?
///   - Is it a recognized platform (Tenor/Giphy/Discord/direct)?
///   - If direct URL, does it have a known media extension?
///
/// For Tenor/Giphy page URLs, we can't know the media type without fetching
/// the page — returns valid with mediaType=null.
class UrlValidator {
  /// Validate a URL without network access.
  static ValidationResult validate(String url) {
    final trimmed = url.trim();
    if (trimmed.isEmpty) {
      return const ValidationResult.invalid('URL is empty');
    }

    // Parse URL
    final uri = Uri.tryParse(trimmed);
    if (uri == null || !uri.hasScheme) {
      return const ValidationResult.invalid('Invalid URL');
    }

    if (!{'http', 'https'}.contains(uri.scheme)) {
      return ValidationResult.invalid('Unsupported scheme: ${uri.scheme}');
    }

    final host = uri.host.toLowerCase();

    // Tenor — page URL or direct media URL
    if (host.contains('tenor.com')) {
      return ValidationResult.valid(
        platform: SourcePlatform.tenor,
        mediaType: null, // can't know without fetching the page
      );
    }

    // Giphy — page URL or direct media URL
    if (host.contains('giphy.com')) {
      return ValidationResult.valid(
        platform: SourcePlatform.giphy,
        mediaType: null,
      );
    }

    // Discord — CDN URLs
    if (host.contains('discordapp.com') || host.contains('discordapp.net')) {
      if (uri.path.contains('/attachments/')) {
        // Try to detect media type from extension
        final ext = _extensionFromPath(uri.path);
        return ValidationResult.valid(
          platform: SourcePlatform.discord,
          mediaType: MediaType.fromExtension(ext),
        );
      }
    }

    // Direct media URL — check extension
    final ext = _extensionFromPath(uri.path);
    if (ext.isNotEmpty) {
      final mediaType = MediaType.fromExtension(ext);
      if (mediaType != null) {
        return ValidationResult.valid(
          platform: SourcePlatform.direct,
          mediaType: mediaType,
        );
      }
    }

    // Unknown — could be a page URL we don't recognize, or a direct link
    // without a media extension. Allow it but flag as unknown.
    return const ValidationResult.valid(
      platform: SourcePlatform.unknown,
      mediaType: null,
    );
  }

  /// Extract file extension from a URL path (e.g. '/foo/bar.gif' → '.gif').
  static String _extensionFromPath(String path) {
    final cleanPath = path.split('?').first.split('#').first;
    final dotIndex = cleanPath.lastIndexOf('.');
    if (dotIndex == -1 || dotIndex == cleanPath.length - 1) return '';
    final ext = cleanPath.substring(dotIndex).toLowerCase();
    // Sanity check — extensions are short
    if (ext.length > 6) return '';
    return ext;
  }
}

// ─── URL Normalizer ─────────────────────────────────────────────────────────

/// Result of URL normalization — classifies the URL and extracts the
/// direct media URL (which may differ from the page URL for Tenor/Giphy).
class NormalizedUrl {
  /// The direct media URL (the actual GIF/MP4/WebP file).
  /// For Tenor/Giphy page URLs, this is extracted from og:image/og:video meta tags.
  /// For direct URLs, this is the same as the input.
  final String directUrl;

  /// The page URL if known (e.g. tenor.com/view/...).
  /// Null for direct media URLs.
  final String? pageUrl;

  /// Which platform the URL came from.
  final SourcePlatform platform;

  /// Dedup key. Format depends on platform:
  ///   tenor:tenor:{id}  (if we could extract the ID)
  ///   giphy:giphy:{id}
  ///   discord:sha256(url)
  ///   direct:sha256(url)
  ///   unknown:sha256(url)
  final String normalizedId;

  /// Whether this URL should be auto-pinned (downloaded immediately).
  /// Discord URLs expire after ~24h, so they're always pinned.
  final bool autoPin;

  const NormalizedUrl({
    required this.directUrl,
    required this.pageUrl,
    required this.platform,
    required this.normalizedId,
    required this.autoPin,
  });
}

/// Parses and classifies URLs. Fetches HTML for Tenor/Giphy page URLs
/// to extract the direct media URL via og:image/og:video meta tags.
class UrlNormalizer {
  final Dio _dio;

  UrlNormalizer({Dio? dio}) : _dio = dio ?? Dio();

  /// Normalize a user-pasted URL.
  ///
  /// Handles:
  ///   - tenor.com/view/{slug}-{id} → fetch HTML → extract og:image → media.tenor.com/...
  ///   - media.tenor.com/... → direct, no fetch needed
  ///   - giphy.com/gifs/{slug}-{id} → fetch HTML → extract og:image → media.giphy.com/...
  ///   - media.giphy.com/media/{id}/... → direct, no fetch needed
  ///   - cdn.discordapp.com/attachments/... or media.discordapp.net/attachments/... → direct, auto-pin
  ///   - Any other URL ending in .gif/.mp4/.webp/.png/.jpg → direct
  ///   - Anything else → unknown, treat as direct
  Future<NormalizedUrl> normalize(String input) async {
    final url = input.trim();

    // ─── Tenor ────────────────────────────────────────────────────────────
    // Page URL: tenor.com/view/{slug}-{id}
    if (RegExp(r'^https?://tenor\.com/view/').hasMatch(url)) {
      return await _normalizeTenorPage(url) ??
          _fallback(url, SourcePlatform.tenor);
    }
    // Direct media URL: media.tenor.com/...
    if (RegExp(r'^https?://media\.tenor\.com/').hasMatch(url)) {
      return NormalizedUrl(
        directUrl: url,
        pageUrl: null,
        platform: SourcePlatform.tenor,
        normalizedId: 'tenor:${_sha256(url)}',
        autoPin: false,
      );
    }

    // ─── Giphy ────────────────────────────────────────────────────────────
    // Page URL: giphy.com/gifs/{slug}-{id}
    if (RegExp(r'^https?://giphy\.com/gifs/').hasMatch(url)) {
      return await _normalizeGiphyPage(url) ??
          _fallback(url, SourcePlatform.giphy);
    }
    // Direct media URL: media.giphy.com/media/{id}/...
    if (RegExp(r'^https?://media\d*\.giphy\.com/media/').hasMatch(url)) {
      return NormalizedUrl(
        directUrl: url,
        pageUrl: null,
        platform: SourcePlatform.giphy,
        normalizedId: 'giphy:${_sha256(url)}',
        autoPin: false,
      );
    }

    // ─── Discord ──────────────────────────────────────────────────────────
    // cdn.discordapp.com/attachments/... or media.discordapp.net/attachments/...
    // These URLs may be signed/expiring — auto-pin so we download immediately.
    // Strip query parameters for the stored URL (cleaner, Discord resolves
    // internally). Keep full URL for download.
    if (RegExp(r'^https?://(cdn\.discordapp\.com|media\.discordapp\.net)/attachments/')
        .hasMatch(url)) {
      // Strip query params for stored URL
      final uri = Uri.parse(url);
      final cleanUrl = uri.origin + uri.path;

      // Validate the URL is accessible (HEAD request)
      try {
        final response = await _dio.head<dynamic>(url);
        if (response.statusCode != null && response.statusCode! >= 400) {
          return NormalizedUrl(
            directUrl: url, // keep full URL for download attempt
            pageUrl: null,
            platform: SourcePlatform.discord,
            normalizedId: 'discord:${_sha256(cleanUrl)}',
            autoPin: true,
          );
        }
      } catch (_) {
        // HEAD failed — URL might be expired. Still add it (user can retry).
      }

      return NormalizedUrl(
        directUrl: url, // full URL with params for download
        pageUrl: null,
        platform: SourcePlatform.discord,
        normalizedId: 'discord:${_sha256(cleanUrl)}',
        autoPin: true, // Discord URLs expire — always pin
      );
    }

    // ─── Twitter/X ────────────────────────────────────────────────────────
    // Matches: x.com/{user}/status/{id}, twitter.com/{user}/status/{id},
    //          fixupx.com/{user}/status/{id}, fixvx.com/{user}/status/{id},
    //          vxtwitter.com/{user}/status/{id}
    // Uses the fxtwitter API to resolve to direct media URLs.
    final twitterMatch = RegExp(
      r'^https?://(?:www\.)?(?:fixup|fixv|vx)?(?:x|twitter)\.com/\w+/status/(\d+)',
    ).firstMatch(url);
    if (twitterMatch != null) {
      return await _normalizeTwitterUrl(url, twitterMatch.group(1)!) ??
          _fallback(url, SourcePlatform.unknown);
    }

    // ─── Direct media URLs ────────────────────────────────────────────────
    // Anything ending in a known media extension
    if (RegExp(r'\.(gif|mp4|webp|png|jpg|jpeg|webm)(\?|$)', caseSensitive: false)
        .hasMatch(url)) {
      return NormalizedUrl(
        directUrl: url,
        pageUrl: null,
        platform: SourcePlatform.direct,
        normalizedId: 'direct:${_sha256(url)}',
        autoPin: false,
      );
    }

    // ─── Unknown ──────────────────────────────────────────────────────────
    // Treat as direct but flag as unknown
    return NormalizedUrl(
      directUrl: url,
      pageUrl: null,
      platform: SourcePlatform.unknown,
      normalizedId: 'unknown:${_sha256(url)}',
      autoPin: false,
    );
  }

  /// Fetch Tenor page HTML, extract og:image or og:video meta tag.
  Future<NormalizedUrl?> _normalizeTenorPage(String pageUrl) async {
    try {
      final response = await _dio.get<String>(pageUrl,
          options: Options(responseType: ResponseType.json));
      final html = response.data;
      if (html == null) return null;

      final doc = html_parser.parse(html);
      final ogImage = doc
          .querySelector('meta[property="og:image"]')
          ?.attributes['content'];
      final ogVideo = doc
          .querySelector('meta[property="og:video"]')
          ?.attributes['content'];

      final directUrl = ogVideo ?? ogImage;
      if (directUrl == null) return null;

      // Try to extract the Tenor ID from the page URL
      final idMatch =
          RegExp(r'tenor\.com/view/[^-]+-(\d+)').firstMatch(pageUrl);
      final tenorId = idMatch?.group(1);

      return NormalizedUrl(
        directUrl: directUrl,
        pageUrl: pageUrl,
        platform: SourcePlatform.tenor,
        normalizedId: tenorId != null ? 'tenor:$tenorId' : 'tenor:${_sha256(pageUrl)}',
        autoPin: false,
      );
    } catch (_) {
      return null;
    }
  }

  /// Fetch Giphy page HTML, extract og:image meta tag.
  Future<NormalizedUrl?> _normalizeGiphyPage(String pageUrl) async {
    try {
      final response = await _dio.get<String>(pageUrl,
          options: Options(responseType: ResponseType.json));
      final html = response.data;
      if (html == null) return null;

      final doc = html_parser.parse(html);
      final ogImage = doc
          .querySelector('meta[property="og:image"]')
          ?.attributes['content'];

      if (ogImage == null) return null;

      // Extract Giphy ID from URL: giphy.com/gifs/{slug}-{id}
      final idMatch =
          RegExp(r'giphy\.com/gifs/(?:.*-)?([a-zA-Z0-9]+)$').firstMatch(pageUrl);
      final giphyId = idMatch?.group(1);

      return NormalizedUrl(
        directUrl: ogImage,
        pageUrl: pageUrl,
        platform: SourcePlatform.giphy,
        normalizedId: giphyId != null ? 'giphy:$giphyId' : 'giphy:${_sha256(pageUrl)}',
        autoPin: false,
      );
    } catch (_) {
      return null;
    }
  }

  /// Resolve a Twitter/X URL via the fxtwitter API.
  ///
  /// Calls https://api.fxtwitter.com/status/{id}/en which returns JSON
  /// with media URLs (videos as MP4, GIFs as direct media URLs).
  Future<NormalizedUrl?> _normalizeTwitterUrl(
      String pageUrl, String statusId) async {
    try {
      final response = await _dio.get<Map<String, dynamic>>(
        'https://api.fxtwitter.com/status/$statusId/en',
      );
      final tweet = response.data?['tweet'] as Map<String, dynamic>?;
      if (tweet == null) return null;

      final media = tweet['media'] as Map<String, dynamic>?;
      if (media == null) return null;

      final photos = media['photos'] as List<dynamic>?;
      final videos = media['videos'] as List<dynamic>?;

      String? directUrl;
      if (videos != null && videos.isNotEmpty) {
        // Prefer video — usually GIFs converted to MP4 on Twitter
        final video = videos[0] as Map<String, dynamic>;
        directUrl = video['url'] as String?;
      } else if (photos != null && photos.isNotEmpty) {
        final photo = photos[0] as Map<String, dynamic>;
        directUrl = photo['url'] as String?;
      }

      if (directUrl == null) return null;

      // Also grab the tweet text for potential use as title
      final text = tweet['text'] as String?;

      return NormalizedUrl(
        directUrl: directUrl,
        pageUrl: pageUrl,
        platform: SourcePlatform.direct,
        normalizedId: 'twitter:$statusId',
        autoPin: false,
      );
    } catch (_) {
      return null;
    }
  }

  /// Fallback when page URL parsing fails — use the original URL as direct.
  NormalizedUrl _fallback(String url, SourcePlatform platform) {
    return NormalizedUrl(
      directUrl: url,
      pageUrl: null,
      platform: platform,
      normalizedId: '${platform.name}:${_sha256(url)}',
      autoPin: platform == SourcePlatform.discord,
    );
  }

  /// Simple SHA256 hash for URL dedup. Uses crypto package if available,
  /// otherwise a basic hash. We don't need cryptographic strength here —
  /// just a consistent identifier.
  ///
  /// TODO: use package:crypto for real SHA256. For now, a simple hash
  /// that's deterministic but not collision-resistant. Fine for dedup
  /// at our scale (thousands, not millions).
  static String _sha256(String input) {
    // Simple FNV-1a hash — deterministic, fast, good enough for dedup
    var hash = 0xcbf29ce484222325;
    for (var i = 0; i < input.length; i++) {
      hash ^= input.codeUnitAt(i);
      hash = (hash * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF;
    }
    return hash.toRadixString(16);
  }
}

// ─── FFmpeg Wrapper ─────────────────────────────────────────────────────────

/// Media info from ffprobe.
class MediaInfo {
  final String? mimeType;
  final int? width;
  final int? height;
  final int? duration;
  final int fileSizeBytes;
  final bool isAnimated;

  const MediaInfo({
    this.mimeType,
    this.width,
    this.height,
    this.duration,
    required this.fileSizeBytes,
    this.isAnimated = false,
  });
}

/// Wraps ffmpeg/ffprobe subprocess calls for thumbnail generation and probing.
///
/// Requires ffmpeg + ffprobe on PATH (or bundled in assets/bin/).
/// Detection happens at startup; if missing, thumbnail generation is skipped
/// (graceful degradation — the app still works, just no thumbnails).
class FfmpegWrapper {
  final String ffmpegPath;
  final String ffprobePath;

  FfmpegWrapper({required this.ffmpegPath, required this.ffprobePath});

  /// Detect ffmpeg + ffprobe on PATH. Returns null if not found.
  ///
  /// On Windows: uses `where ffmpeg` / `where ffprobe`.
  /// On POSIX: uses `which ffmpeg` / `which ffprobe`.
  static Future<FfmpegWrapper?> detect() async {
    final whichCmd = Platform.isWindows ? 'where' : 'which';

    try {
      final ffmpegResult =
          await Process.run(whichCmd, ['ffmpeg'], runInShell: true);
      if (ffmpegResult.exitCode != 0) return null;
      final ffmpegPath = (ffmpegResult.stdout as String)
          .split('\n')
          .first
          .trim();

      final ffprobeResult =
          await Process.run(whichCmd, ['ffprobe'], runInShell: true);
      if (ffprobeResult.exitCode != 0) return null;
      final ffprobePath = (ffprobeResult.stdout as String)
          .split('\n')
          .first
          .trim();

      return FfmpegWrapper(ffmpegPath: ffmpegPath, ffprobePath: ffprobePath);
    } catch (_) {
      return null;
    }
  }

  /// Create an FfmpegWrapper from explicit paths (e.g. bundled binaries).
  static FfmpegWrapper fromPaths({
    required String ffmpegPath,
    required String ffprobePath,
  }) {
    return FfmpegWrapper(ffmpegPath: ffmpegPath, ffprobePath: ffprobePath);
  }

  /// Probe a media file for metadata (dimensions, mime, duration, animation).
  Future<MediaInfo> probe(String inputPath) async {
    final result = await Process.run(
      ffprobePath,
      [
        '-v', 'error', // show errors only (not quiet — we want to see failures)
        '-analyzeduration', '10000000',
        '-probesize', '10000000',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        inputPath,
      ],
    );

    if (result.exitCode != 0) {
      print('[ffprobe] FAILED for $inputPath (exit ${result.exitCode}): ${result.stderr}');
      final fileSize = await File(inputPath).length();
      return MediaInfo(fileSizeBytes: fileSize);
    }

    final jsonStr = result.stdout as String;
    if (jsonStr.trim().isEmpty) {
      print('[ffprobe] empty output for $inputPath');
      final fileSize = await File(inputPath).length();
      return MediaInfo(fileSizeBytes: fileSize);
    }

    final Map<String, dynamic> json;
    try {
      json = jsonDecode(jsonStr) as Map<String, dynamic>;
    } catch (e) {
      print('[ffprobe] JSON parse error for $inputPath: $e');
      final fileSize = await File(inputPath).length();
      return MediaInfo(fileSizeBytes: fileSize);
    }

    final streams = json['streams'] as List<dynamic>?;
    final format = json['format'] as Map<String, dynamic>?;

    String? mimeType;
    int? width;
    int? height;
    int? duration;
    var isAnimated = false;

    if (streams != null && streams.isNotEmpty) {
      final stream = streams[0] as Map<String, dynamic>;
      final codecName = stream['codec_name'] as String?;
      width = (stream['width'] as num?)?.toInt();
      height = (stream['height'] as num?)?.toInt();
      duration = (stream['duration'] as num?)?.toInt();

      print('[ffprobe] $inputPath: codec=$codecName, ${width}x$height, '
          'nb_frames=${stream['nb_frames']}, duration=$duration');

      // Guess mime from codec
      if (codecName == 'gif') {
        mimeType = 'image/gif';
        // GIF is animated if it has more than 1 frame (nb_frames > 1)
        final nbFrames = stream['nb_frames'];
        if (nbFrames != null) {
          isAnimated = (nbFrames is num ? nbFrames.toInt() : int.tryParse(nbFrames.toString()) ?? 1) > 1;
        }
      } else if (codecName == 'png') {
        mimeType = 'image/png';
      } else if (codecName == 'mjpeg') {
        mimeType = 'image/jpeg';
      } else if (codecName == 'webp') {
        mimeType = 'image/webp';
      } else if (codecName == 'h264' || codecName == 'hevc') {
        mimeType = 'video/mp4';
        isAnimated = true;
      } else if (codecName == 'vp9' || codecName == 'vp8') {
        mimeType = 'video/webm';
        isAnimated = true;
      }
    } else {
      print('[ffprobe] $inputPath: no streams found. '
          'streams=${streams?.length}, json keys=${json.keys.toList()}');
    }

    final fileSize = (format?['size'] as String?) != null
        ? int.tryParse(format!['size'] as String) ?? await File(inputPath).length()
        : await File(inputPath).length();

    return MediaInfo(
      mimeType: mimeType,
      width: width,
      height: height,
      duration: duration,
      fileSizeBytes: fileSize,
      isAnimated: isAnimated,
    );
  }

  /// Generate a static WebP thumbnail from any input (image, gif, video).
  ///
  /// Uses the first frame for animated inputs.
  Future<void> generateStaticThumbnail({
    required String inputPath,
    required String outputPath,
    int maxWidth = 256,
    int quality = 80,
  }) async {
    // ffmpeg CANNOT decode animated WebP as input (encoder-only support).
    // For .webp inputs, use package:image instead.
    if (inputPath.toLowerCase().endsWith('.webp')) {
      await _generateStaticThumbnailViaImageLib(inputPath, outputPath, maxWidth);
      return;
    }

    final result = await Process.run(
      ffmpegPath,
      [
        '-y',
        '-i', inputPath,
        '-vframes', '1',
        '-vf', 'scale=$maxWidth:-1',
        '-c:v', 'libwebp',
        '-quality', quality.toString(),
        outputPath,
      ],
    );

    if (result.exitCode != 0) {
      throw Exception(
          'ffmpeg static thumbnail failed (exit ${result.exitCode}):\n'
          'stderr: ${result.stderr}');
    }
  }

  /// Fallback for WebP inputs using package:image (can decode animated WebP).
  /// Imported lazily to avoid circular deps in core — the app provides this
  /// via a callback. For now, we shell out to a simple approach: try ffmpeg
  /// with -analyzeduration and -probesize increased.
  Future<void> _generateStaticThumbnailViaImageLib(
      String inputPath, String outputPath, int maxWidth) async {
    // Try ffmpeg with increased probe settings first (sometimes helps)
    final result = await Process.run(
      ffmpegPath,
      [
        '-y',
        '-analyzeduration', '10000000',
        '-probesize', '10000000',
        '-i', inputPath,
        '-vframes', '1',
        '-vf', 'scale=$maxWidth:-1',
        '-c:v', 'libwebp',
        '-quality', '80',
        outputPath,
      ],
    );

    if (result.exitCode == 0) return;

    // ffmpeg can't handle it — the app's image provider will display the
    // original file directly (Flutter's Image widget can decode animated WebP).
    // Just copy the original as the "thumbnail" — it'll be larger but works.
    try {
      await File(inputPath).copy(outputPath);
    } catch (e) {
      throw Exception('Cannot create thumbnail for animated WebP: $e');
    }
  }

  Future<void> generateAnimatedThumbnail({
    required String inputPath,
    required String outputPath,
    int maxWidth = 256,
    int fps = 10,
    int quality = 70,
  }) async {
    // ffmpeg can't decode animated WebP — just copy the original.
    // The image provider will display it natively (Flutter supports animated WebP).
    if (inputPath.toLowerCase().endsWith('.webp')) {
      try {
        await File(inputPath).copy(outputPath);
        return;
      } catch (e) {
        throw Exception('Cannot copy animated WebP for thumbnail: $e');
      }
    }

    final result = await Process.run(
      ffmpegPath,
      [
        '-y',
        '-i', inputPath,
        '-vf', 'scale=$maxWidth:-1,fps=$fps',
        '-c:v', 'libwebp',
        '-loop', '0',
        '-quality', quality.toString(),
        outputPath,
      ],
    );

    if (result.exitCode != 0) {
      throw Exception(
          'ffmpeg animated thumbnail failed (exit ${result.exitCode}):\n'
          'stderr: ${result.stderr}');
    }
  }
}

// ─── Ingest Pipeline ────────────────────────────────────────────────────────

/// Events emitted by the ingest pipeline. The Coordinator forwards these
/// to the UI via signals so the grid can update in real time.
sealed class IngestEvent {}

class IngestProgressEvent extends IngestEvent {
  final String reactionId;
  final ReactionStatus status;
  final double progress; // 0.0 – 1.0
  final Reaction? reaction; // full updated reaction (includes width/height/etc from download)

  IngestProgressEvent({
    required this.reactionId,
    required this.status,
    required this.progress,
    this.reaction,
  });
}

class IngestCompleteEvent extends IngestEvent {
  final Reaction reaction; // fully updated reaction

  IngestCompleteEvent(this.reaction);
}

class IngestFailedEvent extends IngestEvent {
  final String reactionId;
  final String error;
  final Reaction? reaction; // the reaction with whatever data was collected before failure

  IngestFailedEvent({
    required this.reactionId,
    required this.error,
    this.reaction,
  });
}

/// Configuration for the ingest pipeline.
class IngestConfig {
  /// Whether to generate animated WebP previews (in addition to static).
  final bool animatedPreviewsEnabled;

  /// Whether to run OCR on thumbnails.
  final bool ocrEnabled;

  /// Whether to compute CLIP image embeddings (cross-modal search).
  final bool imageEmbeddingsEnabled;

  /// Active model versions (for tracking which model produced each derivative).
  final String textModelVersion;
  final String imageModelVersion;
  final String ocrModelVersion;

  const IngestConfig({
    this.animatedPreviewsEnabled = false,
    this.ocrEnabled = false,
    this.imageEmbeddingsEnabled = true,
    this.textModelVersion = 'potion-base-32M',
    this.imageModelVersion = 'clip-vit-b32-fp16-v1',
    this.ocrModelVersion = 'pp-ocr-v5',
  });
}

/// The ingest pipeline — processes a Reaction from queued to ready.
///
/// Each stage:
///   1. Updates the Reaction's status + progress
///   2. Emits an IngestProgressEvent
///   3. Does the work
///   4. Updates derivative file paths on the Reaction
///
/// The pipeline is designed to be resumable: if the app crashes, on next
/// launch the Coordinator re-queues reactions with status NOT IN (ready, failed).
/// Stages are idempotent — re-running just overwrites.
class IngestPipeline {
  final Storage storage;
  final TextEmbedder textEmbedder;
  final ImageEmbedder? imageEmbedder;
  final OcrEngine? ocrEngine;
  final FfmpegWrapper? ffmpeg;
  final Dio dio;
  final IngestConfig config;

  IngestPipeline({
    required this.storage,
    required this.textEmbedder,
    this.imageEmbedder,
    this.ocrEngine,
    this.ffmpeg,
    required this.dio,
    required this.config,
  });

  /// Process a reaction end-to-end. Emits events as it progresses.
  ///
  /// The [initial] reaction should have status=queued. The pipeline returns
  /// the final updated reaction (or throws on failure).
  Stream<IngestEvent> process(Reaction initial) async* {
    var reaction = initial;
    print('[Ingest] Starting pipeline for ${reaction.id} (${reaction.url})');

    try {
      // ─── Stage 1: Download ──────────────────────────────────────────────
      print('[Ingest] ${reaction.id}: downloading...');
      reaction = reaction.copyWith(status: ReactionStatus.downloading, progress: 0.0);
      yield IngestProgressEvent(
          reactionId: reaction.id,
          status: reaction.status,
          progress: reaction.progress);

      final downloadResult = await _download(reaction);
      reaction = downloadResult.reaction;
      print('[Ingest] ${reaction.id}: downloaded ${reaction.fileSizeBytes} bytes, '
          '${reaction.width}x${reaction.height}, mime=${reaction.mimeType}, '
          'animated=${downloadResult.mediaInfo.isAnimated}');
      yield IngestProgressEvent(
          reactionId: reaction.id,
          status: reaction.status,
          progress: 0.4,
          reaction: reaction);

      // ─── Stage 2: Thumbnail ─────────────────────────────────────────────
      print('[Ingest] ${reaction.id}: generating thumbnail...');
      reaction = reaction.copyWith(status: ReactionStatus.thumbnailing, progress: 0.5);
      yield IngestProgressEvent(
          reactionId: reaction.id,
          status: reaction.status,
          progress: reaction.progress,
          reaction: reaction);

      await _generateThumbnails(reaction, downloadResult.localPath);

      final staticThumbPath = storage.thumbnailStaticPath(reaction.id);
      reaction = reaction.copyWith(
        thumbnailStatic: p.relative(staticThumbPath, from: storage.rootPath),
      );
      // Check if the source is inherently animated (regardless of ffprobe detection,
      // which fails on animated WebP and sometimes GIFs)
      final isLikelyAnimated = downloadResult.mediaInfo.isAnimated ||
          reaction.mimeType == 'image/gif' ||
          reaction.mimeType == 'image/webp' ||
          reaction.mimeType == 'video/mp4' ||
          reaction.mimeType == 'video/webm' ||
          reaction.url.toLowerCase().endsWith('.gif') ||
          reaction.url.toLowerCase().endsWith('.webp') ||
          reaction.url.toLowerCase().endsWith('.mp4') ||
          reaction.url.toLowerCase().endsWith('.webm');

      if (config.animatedPreviewsEnabled && isLikelyAnimated) {
        final animThumbPath = storage.thumbnailAnimatedPath(reaction.id);
        reaction = reaction.copyWith(
          thumbnailAnimated: p.relative(animThumbPath, from: storage.rootPath),
        );
      }
      print('[Ingest] ${reaction.id}: thumbnail ready at ${reaction.thumbnailStatic}');
      yield IngestProgressEvent(
          reactionId: reaction.id,
          status: reaction.status,
          progress: 0.6,
          reaction: reaction);

      // ─── Stage 3: Text embedding ───────────────────────────────────────
      print('[Ingest] ${reaction.id}: text embedding...');
      reaction = reaction.copyWith(status: ReactionStatus.embedding, progress: 0.65);
      yield IngestProgressEvent(
          reactionId: reaction.id,
          status: reaction.status,
          progress: reaction.progress,
          reaction: reaction);

      final embeddableText = reaction.embeddableText;
      if (embeddableText.isNotEmpty) {
        final textVec = await textEmbedder.embed(embeddableText);
        final vecPath =
            storage.textEmbeddingPath(reaction.id, 'clip-vit-b32-fp16', config.textModelVersion);
        await writeVectorFile(vecPath, textVec);
        reaction = reaction.copyWith(
          textEmbeddingPath: p.relative(vecPath, from: storage.rootPath),
          textModelVersion: config.textModelVersion,
        );
        print('[Ingest] ${reaction.id}: text embedding done');
      } else {
        print('[Ingest] ${reaction.id}: no text to embed (no title/tags/ocr)');
      }
      yield IngestProgressEvent(
          reactionId: reaction.id,
          status: reaction.status,
          progress: 0.75);

      // ─── Stage 4: Image embedding ──────────────────────────────────────
      print('[Ingest] ${reaction.id}: image embedding...');
      if (config.imageEmbeddingsEnabled && imageEmbedder != null) {
        try {
          await imageEmbedder!.init();
          final thumbAbsPath = p.join(storage.rootPath, reaction.thumbnailStatic!);
          final imageVec = await imageEmbedder!.embedFile(thumbAbsPath);
          final vecPath = storage.imageEmbeddingPath(
              reaction.id, 'clip-vit-b32-fp16', config.imageModelVersion);
          await writeVectorFile(vecPath, imageVec);
          reaction = reaction.copyWith(
            imageEmbeddingPath: p.relative(vecPath, from: storage.rootPath),
            imageModelVersion: config.imageModelVersion,
          );
          print('[Ingest] ${reaction.id}: image embedding done');
        } catch (e) {
          print('[Ingest] ${reaction.id}: image embedding FAILED (non-fatal): $e');
        }
      }
      yield IngestProgressEvent(
          reactionId: reaction.id,
          status: reaction.status,
          progress: 0.85,
          reaction: reaction);

      // ─── Stage 5: OCR (optional) ───────────────────────────────────────
      if (config.ocrEnabled && ocrEngine != null) {
        print('[Ingest] ${reaction.id}: OCR...');
        reaction = reaction.copyWith(status: ReactionStatus.ocr, progress: 0.9);
        yield IngestProgressEvent(
            reactionId: reaction.id,
            status: reaction.status,
            progress: reaction.progress,
            reaction: reaction);

        try {
          await ocrEngine!.init();
          final thumbAbsPath = p.join(storage.rootPath, reaction.thumbnailStatic!);
          final ocrText = await ocrEngine!.ocrFile(thumbAbsPath);
          if (ocrText != null && ocrText.isNotEmpty) {
            final ocrPath = storage.ocrPath(reaction.id);
            await File(ocrPath).writeAsString(ocrText, flush: true);
            reaction = reaction.copyWith(
              ocrText: ocrText,
              ocrModelVersion: config.ocrModelVersion,
            );
            print('[Ingest] ${reaction.id}: OCR text: "${ocrText.substring(0, ocrText.length > 50 ? 50 : ocrText.length)}..."');

            final enrichedText = reaction.embeddableText;
            if (enrichedText.isNotEmpty) {
              final textVec = await textEmbedder.embed(enrichedText);
              final vecPath = storage.textEmbeddingPath(
                  reaction.id, 'clip-vit-b32-fp16', config.textModelVersion);
              await writeVectorFile(vecPath, textVec);
            }
          } else {
            print('[Ingest] ${reaction.id}: no text found by OCR');
          }
        } catch (e) {
          print('[Ingest] ${reaction.id}: OCR FAILED (non-fatal): $e');
        }

        try {
          await ocrEngine!.dispose();
        } catch (_) {}
      }

      // ─── Stage 6: Done ──────────────────────────────────────────────────
      reaction = reaction.copyWith(
          status: ReactionStatus.ready, progress: 1.0);
      print('[Ingest] ${reaction.id}: pipeline complete ✓');
      yield IngestCompleteEvent(reaction);
    } catch (e) {
      reaction = reaction.copyWith(
          status: ReactionStatus.failed,
          errorMessage: e.toString());
      yield IngestFailedEvent(
          reactionId: reaction.id,
          error: e.toString(),
          reaction: reaction);
    }
  }

  /// Download the reaction's URL to disk.
  Future<({Reaction reaction, String localPath, MediaInfo mediaInfo})>
      _download(Reaction reaction) async {
    final extension = _guessExtension(reaction.url);
    final downloadPath = storage.downloadPath(reaction.id, extension);
    await Directory(p.dirname(downloadPath)).create(recursive: true);

    // Download to .tmp first, rename on success (crash-safe)
    final tmpPath = '$downloadPath.tmp';
    await dio.download(reaction.url, tmpPath);
    await File(tmpPath).rename(downloadPath);

    // Probe for media info
    MediaInfo mediaInfo;
    if (ffmpeg != null) {
      try {
        mediaInfo = await ffmpeg!.probe(downloadPath);
      } catch (_) {
        final fileSize = await File(downloadPath).length();
        mediaInfo = MediaInfo(fileSizeBytes: fileSize);
      }
    } else {
      final fileSize = await File(downloadPath).length();
      mediaInfo = MediaInfo(fileSizeBytes: fileSize);
    }

    final updated = reaction.copyWith(
      localFile: p.relative(downloadPath, from: storage.rootPath),
      mimeType: mediaInfo.mimeType,
      width: mediaInfo.width,
      height: mediaInfo.height,
      fileSizeBytes: mediaInfo.fileSizeBytes,
    );

    return (reaction: updated, localPath: downloadPath, mediaInfo: mediaInfo);
  }

  /// Generate thumbnails (static + optional animated).
  Future<void> _generateThumbnails(Reaction reaction, String localPath) async {
    if (ffmpeg == null) return; // graceful degradation — no thumbnails

    await Directory(storage.thumbnailsStaticDir).create(recursive: true);
    final staticPath = storage.thumbnailStaticPath(reaction.id);
    await ffmpeg!.generateStaticThumbnail(
      inputPath: localPath,
      outputPath: staticPath,
    );

    if (config.animatedPreviewsEnabled) {
      // Only generate animated thumbnail if the source is animated
      // (we don't know here — the caller should check mediaInfo.isAnimated)
      // For now, try anyway; ffmpeg will just produce a 1-frame WebP for static inputs
      try {
        await Directory(storage.thumbnailsAnimatedDir).create(recursive: true);
        final animPath = storage.thumbnailAnimatedPath(reaction.id);
        await ffmpeg!.generateAnimatedThumbnail(
          inputPath: localPath,
          outputPath: animPath,
        );
      } catch (_) {
        // Animated thumbnail failure is non-fatal
      }
    }
  }

  /// Guess file extension from URL.
  String _guessExtension(String url) {
    final uri = Uri.parse(url);
    final path = uri.path.toLowerCase();
    for (final ext in ['.gif', '.mp4', '.webp', '.png', '.jpg', '.jpeg', '.webm']) {
      if (path.endsWith(ext)) return ext.substring(1); // strip the dot
    }
    return 'bin'; // unknown
  }
}

// ─── Reaction Factory ───────────────────────────────────────────────────────

/// Creates new Reaction entries from URLs.
class ReactionFactory {
  final UrlNormalizer _normalizer;
  final Uuid _uuid;

  ReactionFactory({UrlNormalizer? normalizer, Uuid? uuid})
      : _normalizer = normalizer ?? UrlNormalizer(),
        _uuid = uuid ?? const Uuid();

  /// Create a new Reaction from a URL.
  ///
  /// Returns null if the URL is a duplicate (already in metadata).
  Future<Reaction?> create({
    required String url,
    String? title,
    List<String>? tags,
    required List<Reaction> existingReactions,
  }) async {
    final normalized = await _normalizer.normalize(url);

    // Dedup check
    for (final existing in existingReactions) {
      if (existing.urlNormalized == normalized.normalizedId) {
        return null; // duplicate
      }
    }

    return Reaction(
      id: _uuid.v7(),
      url: normalized.directUrl,
      pageUrl: normalized.pageUrl,
      urlNormalized: normalized.normalizedId,
      sourcePlatform: normalized.platform,
      addedAt: DateTime.now(),
      status: ReactionStatus.queued,
      title: title,
      tags: tags ?? const [],
      pinned: normalized.autoPin,
    );
  }
}
