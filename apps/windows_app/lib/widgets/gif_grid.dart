import 'dart:io';
import 'package:flutter/material.dart';
import 'package:forui/forui.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../models/gif_media.dart';
import '../state/signals.dart';
import '../services/clipboard_service.dart';

class GifGrid extends StatelessWidget {
  const GifGrid({super.key});

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        // Empty query -> hotlist; otherwise search results.
        final query = searchQuery.value;
        final items = query.isEmpty ? recentUsage.value : searchResults.value;

        if (items.isEmpty && query.isNotEmpty) {
          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              spacing: 8,
              children: [const Icon(FLucideIcons.searchX, size: 32), Text(isSearching.value ? 'Searching…' : 'No matches')],
            ),
          );
        }

        if (items.isEmpty) {
          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              spacing: 12,
              children: [
                const Icon(FLucideIcons.imagePlus, size: 40),
                const Text('Your library is empty'),
                FButton(
                  mainAxisSize: MainAxisSize.min,
                  onPress: () {
                    // TODO: trigger import flow
                  },
                  child: const Text('Add your first reaction'),
                ),
              ],
            ),
          );
        }

        return GridView.builder(
          padding: const EdgeInsets.all(8),
          gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
            maxCrossAxisExtent: 180,
            mainAxisSpacing: 8,
            crossAxisSpacing: 8,
            childAspectRatio: 1.0,
          ),
          itemCount: items.length,
          itemBuilder: (context, i) => _GifTile(item: items[i]),
        );
      },
    );
  }
}

class _GifTile extends StatelessWidget {
  final GifMedia item;
  const _GifTile({required this.item});

  @override
  Widget build(BuildContext context) {
    final colors = context.theme.colors;
    return FCard.raw(
      clipBehavior: Clip.antiAlias, // was: Clip.antebugex (typo)
      child: InkWell(
        onTap: () => ClipboardService.copyReaction(item),
        onLongPress: () {
          // TODO: long-press -> "add context" popover (smart haptic tagging)
        },
        child: Stack(
          fit: StackFit.expand,
          children: [
            item.thumbnailPath != null
                ? Image.file(File(item.thumbnailPath!), fit: BoxFit.cover)
                : Container(
                    color: colors.muted,
                    child: Icon(FLucideIcons.image, size: 28, color: colors.mutedForeground),
                  ),
            Positioned(
              right: 4,
              bottom: 4,
              child: FBadge(variant: FBadgeVariant.secondary, child: Text('${item.usageCount}')),
            ),
          ],
        ),
      ),
    );
  }
}
