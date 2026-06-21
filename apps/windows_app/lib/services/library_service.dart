// lib/services/library_service.dart
import '../models/gif_media.dart';
import '../state/signals.dart';

class LibraryService {
  /// TODO: SELECT * FROM gif_media ORDER BY usage_count DESC LIMIT 60
  static Future<void> loadRecent() async {
    recentUsage.value = []; // stub
  }
}
