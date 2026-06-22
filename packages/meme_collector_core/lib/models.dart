/// Data shapes for the entire meme_collector_core package.
///
/// No logic, just types + JSON codec. Everything else in the package
/// depends on these.
library;

import 'package:meta/meta.dart';

/// Pipeline state for a reaction. Order roughly = ingest pipeline order.
enum ReactionStatus {
  /// Just added, not yet downloading.
  queued,

  /// Network fetch in progress.
  downloading,

  /// File on disk, generating thumbnails.
  thumbnailing,

  /// Computing text + image embeddings.
  embedding,

  /// Running OCR (optional stage, only if config.ocrEnabled).
  ocr,

  /// Fully processed, searchable, ready to click.
  ready,

  /// Something went wrong. See [Reaction.errorMessage].
  failed;

  static ReactionStatus fromString(String s) =>
      ReactionStatus.values.firstWhere((e) => e.name == s,
          orElse: () => ReactionStatus.queued);

  String toJson() => name;
}

/// Where the URL came from. Affects dedup logic and pinning policy.
enum SourcePlatform {
  /// tenor.com/view/{slug}-{id} or media.tenor.com/...
  tenor,

  /// giphy.com/gifs/{slug}-{id} or media.giphy.com/media/{id}/...
  giphy,

  /// cdn.discordapp.com/attachments/... or media.discordapp.net/attachments/...
  /// Discord URLs may be signed/expiring — auto-pinned on import.
  discord,

  /// A direct media URL (no platform-specific page). Dedup by sha256(url).
  direct,

  /// Anything we don't recognize. Treated like [direct] but flagged.
  unknown;

  static SourcePlatform fromString(String s) =>
      SourcePlatform.values.firstWhere((e) => e.name == s,
          orElse: () => SourcePlatform.unknown);

  String toJson() => name;
}

/// A single reaction entry.
///
/// Identity is [id] (UUID v7, time-sortable). The URL is the "real" identity
/// from the user's perspective (Discord-style: click to copy URL, file is
/// just a preview derivative). Files on disk (downloads, thumbnails,
/// embeddings, OCR) are all named by [id] and live under the storage root.
@immutable
class Reaction {
  final String id;
  final String url;
  final String? pageUrl;
  final String urlNormalized;
  final SourcePlatform sourcePlatform;
  final DateTime addedAt;

  /// Current pipeline state. `ready` = fully searchable + clickable.
  final ReactionStatus status;

  /// 0.0–1.0 during ingest. 1.0 when ready.
  final double progress;

  /// Set when [status] == [ReactionStatus.failed].
  final String? errorMessage;

  // ─── User metadata ─────────────────────────────────────────────────────
  /// User-supplied title. Used for text embedding if non-empty.
  final String? title;

  /// User-supplied tags. Concatenated into the text embedding.
  final List<String> tags;

  /// Free-form notes. Not embedded (could contain personal context).
  final String? notes;

  // ─── Usage ─────────────────────────────────────────────────────────────
  final int usageCount;
  final DateTime? lastUsedAt;

  // ─── Pinning (dead-URL resilience) ─────────────────────────────────────
  /// If true, the local file is kept forever even if URL dies.
  /// Discord imports auto-pin on add (signed URLs expire).
  final bool pinned;

  /// null = not yet checked. true = HEAD returned 4xx/5xx.
  final bool? urlDead;

  // ─── Derivative file paths (relative to storage root, null = not generated) ─
  /// Original downloaded file. Null until download completes.
  /// Path: `downloads/{id}.{ext}`
  final String? localFile;

  /// Static WebP thumbnail. Always generated once download completes.
  /// Path: `thumbnails_static/{id}.webp`
  final String? thumbnailStatic;

  /// Animated WebP thumbnail (only if config.animatedPreviewsEnabled).
  /// Path: `thumbnails_animated/{id}.webp`
  final String? thumbnailAnimated;

  /// Raw OCR text (only if config.ocrEnabled).
  /// Path: `ocr/{id}.txt`
  final String? ocrText;

  /// Float32 vector file (512-dim model2vec).
  /// Path: `embeddings_text/{producer}__{version}/{id}.f32.bin`
  final String? textEmbeddingPath;

  /// Float32 vector file (512-dim CLIP vision).
  /// Path: `embeddings_image/{producer}__{version}/{id}.f32.bin`
  final String? imageEmbeddingPath;

  // ─── Model versions (for selective re-processing) ──────────────────────
  /// e.g. 'potion-base-32M'. When model2vec upgrades, re-embed selectively.
  final String? textModelVersion;

  /// e.g. 'clip-vit-b32-int8-v1'.
  final String? imageModelVersion;

  /// e.g. 'pp-ocr-v5'.
  final String? ocrModelVersion;

  // ─── File metadata (filled in after download) ──────────────────────────
  final String? mimeType;
  final int? width;
  final int? height;
  final int? fileSizeBytes;

  const Reaction({
    required this.id,
    required this.url,
    required this.urlNormalized,
    required this.sourcePlatform,
    required this.addedAt,
    this.status = ReactionStatus.queued,
    this.progress = 0.0,
    this.errorMessage,
    this.title,
    this.tags = const [],
    this.notes,
    this.usageCount = 0,
    this.lastUsedAt,
    this.pinned = false,
    this.urlDead,
    this.localFile,
    this.thumbnailStatic,
    this.thumbnailAnimated,
    this.ocrText,
    this.textEmbeddingPath,
    this.imageEmbeddingPath,
    this.textModelVersion,
    this.imageModelVersion,
    this.ocrModelVersion,
    this.mimeType,
    this.width,
    this.height,
    this.fileSizeBytes,
    this.pageUrl,
  });

  /// Combined text used for embedding. Empty if no metadata yet.
  String get embeddableText {
    final parts = <String>[
      if (title != null && title!.isNotEmpty) title!,
      ...tags,
      if (ocrText != null && ocrText!.isNotEmpty) ocrText!,
    ];
    return parts.join(' ');
  }

  /// True if at least the text embedding exists (searchable by text).
  bool get isSearchable => textEmbeddingPath != null;

  /// True if the original file is on disk (can be copied as bytes).
  bool get hasLocalFile => localFile != null;

  Reaction copyWith({
    String? url,
    String? pageUrl,
    ReactionStatus? status,
    double? progress,
    String? errorMessage,
    String? title,
    List<String>? tags,
    String? notes,
    int? usageCount,
    DateTime? lastUsedAt,
    bool? pinned,
    bool? urlDead,
    String? localFile,
    String? thumbnailStatic,
    String? thumbnailAnimated,
    String? ocrText,
    String? textEmbeddingPath,
    String? imageEmbeddingPath,
    String? textModelVersion,
    String? imageModelVersion,
    String? ocrModelVersion,
    String? mimeType,
    int? width,
    int? height,
    int? fileSizeBytes,
  }) {
    return Reaction(
      id: id,
      url: url ?? this.url,
      urlNormalized: urlNormalized,
      sourcePlatform: sourcePlatform,
      addedAt: addedAt,
      status: status ?? this.status,
      progress: progress ?? this.progress,
      errorMessage: errorMessage,
      title: title ?? this.title,
      tags: tags ?? this.tags,
      notes: notes ?? this.notes,
      usageCount: usageCount ?? this.usageCount,
      lastUsedAt: lastUsedAt ?? this.lastUsedAt,
      pinned: pinned ?? this.pinned,
      urlDead: urlDead ?? this.urlDead,
      localFile: localFile ?? this.localFile,
      thumbnailStatic: thumbnailStatic ?? this.thumbnailStatic,
      thumbnailAnimated: thumbnailAnimated ?? this.thumbnailAnimated,
      ocrText: ocrText ?? this.ocrText,
      textEmbeddingPath: textEmbeddingPath ?? this.textEmbeddingPath,
      imageEmbeddingPath: imageEmbeddingPath ?? this.imageEmbeddingPath,
      textModelVersion: textModelVersion ?? this.textModelVersion,
      imageModelVersion: imageModelVersion ?? this.imageModelVersion,
      ocrModelVersion: ocrModelVersion ?? this.ocrModelVersion,
      mimeType: mimeType ?? this.mimeType,
      width: width ?? this.width,
      height: height ?? this.height,
      fileSizeBytes: fileSizeBytes ?? this.fileSizeBytes,
      pageUrl: pageUrl ?? this.pageUrl,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'url': url,
        if (pageUrl != null) 'pageUrl': pageUrl,
        'urlNormalized': urlNormalized,
        'sourcePlatform': sourcePlatform.toJson(),
        'addedAt': addedAt.toIso8601String(),
        'status': status.toJson(),
        'progress': progress,
        if (errorMessage != null) 'errorMessage': errorMessage,
        if (title != null) 'title': title,
        'tags': tags,
        if (notes != null) 'notes': notes,
        'usageCount': usageCount,
        if (lastUsedAt != null)
          'lastUsedAt': lastUsedAt!.toIso8601String(),
        'pinned': pinned,
        if (urlDead != null) 'urlDead': urlDead,
        if (localFile != null) 'localFile': localFile,
        if (thumbnailStatic != null) 'thumbnailStatic': thumbnailStatic,
        if (thumbnailAnimated != null)
          'thumbnailAnimated': thumbnailAnimated,
        if (ocrText != null) 'ocrText': ocrText,
        if (textEmbeddingPath != null)
          'textEmbeddingPath': textEmbeddingPath,
        if (imageEmbeddingPath != null)
          'imageEmbeddingPath': imageEmbeddingPath,
        if (textModelVersion != null)
          'textModelVersion': textModelVersion,
        if (imageModelVersion != null)
          'imageModelVersion': imageModelVersion,
        if (ocrModelVersion != null) 'ocrModelVersion': ocrModelVersion,
        if (mimeType != null) 'mimeType': mimeType,
        if (width != null) 'width': width,
        if (height != null) 'height': height,
        if (fileSizeBytes != null) 'fileSizeBytes': fileSizeBytes,
      };

  factory Reaction.fromJson(Map<String, dynamic> json) {
    return Reaction(
      id: json['id'] as String,
      url: json['url'] as String,
      pageUrl: json['pageUrl'] as String?,
      urlNormalized: json['urlNormalized'] as String,
      sourcePlatform:
          SourcePlatform.fromString(json['sourcePlatform'] as String? ?? ''),
      addedAt: DateTime.parse(json['addedAt'] as String),
      status: ReactionStatus.fromString(json['status'] as String? ?? 'queued'),
      progress: (json['progress'] as num?)?.toDouble() ?? 0.0,
      errorMessage: json['errorMessage'] as String?,
      title: json['title'] as String?,
      tags: (json['tags'] as List<dynamic>? ?? [])
          .map((e) => e as String)
          .toList(),
      notes: json['notes'] as String?,
      usageCount: (json['usageCount'] as num?)?.toInt() ?? 0,
      lastUsedAt: json['lastUsedAt'] != null
          ? DateTime.parse(json['lastUsedAt'] as String)
          : null,
      pinned: (json['pinned'] as bool?) ?? false,
      urlDead: json['urlDead'] as bool?,
      localFile: json['localFile'] as String?,
      thumbnailStatic: json['thumbnailStatic'] as String?,
      thumbnailAnimated: json['thumbnailAnimated'] as String?,
      ocrText: json['ocrText'] as String?,
      textEmbeddingPath: json['textEmbeddingPath'] as String?,
      imageEmbeddingPath: json['imageEmbeddingPath'] as String?,
      textModelVersion: json['textModelVersion'] as String?,
      imageModelVersion: json['imageModelVersion'] as String?,
      ocrModelVersion: json['ocrModelVersion'] as String?,
      mimeType: json['mimeType'] as String?,
      width: json['width'] as int?,
      height: json['height'] as int?,
      fileSizeBytes: json['fileSizeBytes'] as int?,
    );
  }
}

/// The on-disk metadata.json shape.
class Metadata {
  /// Bump when the JSON shape changes. Old versions can be migrated.
  static const int currentSchemaVersion = 1;

  final int schemaVersion;

  /// Active model versions. Maps derivative kind → version string.
  /// e.g. {'text_embedding': 'potion-base-32M', 'image_embedding': 'clip-vit-b32-int8-v1', 'ocr': 'pp-ocr-v5'}
  /// When a model upgrades, re-generate derivatives and update this map.
  final Map<String, String> activeModels;

  final List<Reaction> reactions;

  const Metadata({
    this.schemaVersion = currentSchemaVersion,
    required this.activeModels,
    required this.reactions,
  });

  Metadata copyWith({
    Map<String, String>? activeModels,
    List<Reaction>? reactions,
  }) =>
      Metadata(
        schemaVersion: schemaVersion,
        activeModels: activeModels ?? this.activeModels,
        reactions: reactions ?? this.reactions,
      );

  Map<String, dynamic> toJson() => {
        'schemaVersion': schemaVersion,
        'activeModels': activeModels,
        'reactions': reactions.map((r) => r.toJson()).toList(),
      };

  factory Metadata.fromJson(Map<String, dynamic> json) {
    return Metadata(
      schemaVersion: (json['schemaVersion'] as num?)?.toInt() ??
          currentSchemaVersion,
      activeModels:
          (json['activeModels'] as Map<String, dynamic>?)?.cast() ?? {},
      reactions: (json['reactions'] as List<dynamic>? ?? [])
          .map((e) => Reaction.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }

  /// Empty state for first launch.
  factory Metadata.empty() => const Metadata(
        activeModels: {
          'text_embedding': 'potion-base-32M',
          'image_embedding': 'clip-vit-b32-int8-v1',
          'ocr': 'pp-ocr-v5',
        },
        reactions: [],
      );

  /// Find a reaction by ID, or null.
  Reaction? byId(String id) {
    for (final r in reactions) {
      if (r.id == id) return r;
    }
    return null;
  }

  /// Find a reaction by normalized URL (dedup check on add).
  Reaction? byNormalizedUrl(String urlNormalized) {
    for (final r in reactions) {
      if (r.urlNormalized == urlNormalized) return r;
    }
    return null;
  }
}
