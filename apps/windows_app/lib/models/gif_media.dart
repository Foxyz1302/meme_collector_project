class GifMedia {
  final int id;
  final String filePath; // local path to the gif/mp4
  final String? sourceUrl; // original tenor/giphy/discord url
  final String? thumbnailPath;
  final List<String> tags;
  final int usageCount;

  const GifMedia({required this.id, required this.filePath, this.sourceUrl, this.thumbnailPath, this.tags = const [], this.usageCount = 0});

  // TODO: replace once the SQLite schema from the prior step is wired.
  factory GifMedia.fromMap(Map<String, dynamic> m) => GifMedia(
    id: m['id'] as int,
    filePath: m['file_path'] as String,
    sourceUrl: m['source_url'] as String?,
    thumbnailPath: m['thumbnail_path'] as String?,
    tags: const [],
    usageCount: (m['usage_count'] as int?) ?? 0,
  );
}
