import 'package:signals_flutter/signals_flutter.dart';
import '../models/gif_media.dart';

/// Reactive state. Watch() these in widgets for granular rebuilds.
final searchQuery = signal('');
final isSearching = signal(false);
final searchResults = signal<List<GifMedia>>([]);
final recentUsage = signal<List<GifMedia>>([]);
final selectedNav = signal('all'); // 'all' | 'recent' | 'tags' | 'sources'
