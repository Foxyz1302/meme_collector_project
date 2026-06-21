// lib/services/search_service.dart
import '../models/gif_media.dart';
import '../state/signals.dart';

class SearchService {
  /// TODO: implement RRF fusion of FTS5 + visual vector + text vector.
  /// For now: empty query -> hotlist, else return [] (stub).
  static Future<List<GifMedia>> performSearch(String query) async {
    if (query.isEmpty) return recentUsage.value;
    return [];
  }
}
