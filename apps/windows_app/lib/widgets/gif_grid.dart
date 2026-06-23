// ignore_for_file: prefer_const_constructors

import 'package:exui/exui.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:forui/forui.dart';
import 'package:signals_flutter/signals_flutter.dart';
import 'package:waterfall_flow/waterfall_flow.dart';
import 'package:persistent_header_adaptive/persistent_header_adaptive.dart';

import 'package:meme_collector_core/meme_collector_core.dart';
import 'package:windows_app/widgets/app_searchbar.dart';

import '../reaction_image_provider.dart';
import '../state/signals.dart';

class GifGrid extends StatelessWidget {
  const GifGrid({super.key});

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        final query = searchQuery.value;
        final allItems = allReactions.value;
        final searchItems = query.isEmpty ? <SearchResult>[] : searchResults.value;

        final bool isSearching = query.isNotEmpty;
        final items = isSearching ? searchItems : allItems;
        final bool isEmpty = items.isEmpty;

        // Build the content (waterfall or empty)
        Widget contentSliver;
        if (isEmpty) {
          contentSliver = SliverFillRemaining(
            child: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                spacing: 12,
                children: [
                  Icon(isSearching ? FLucideIcons.searchX : FLucideIcons.imagePlus, size: isSearching ? 32 : 40),
                  Text(isSearching ? 'No matches' : 'Your library is empty'),
                  if (!isSearching && allItems.isEmpty)
                    FButton(
                      mainAxisSize: MainAxisSize.min,
                      onPress: () async {
                        final clipData = await Clipboard.getData('text/plain');
                        final url = clipData?.text?.trim();
                        if (url != null && url.isNotEmpty) {
                          await addReaction(url);
                        }
                      },
                      child: const Text('Paste reaction URL'),
                    ),
                ],
              ),
            ),
          );
        } else {
          // Sort for library, keep search order as-is
          final sortedItems = isSearching ? items : (List<Reaction>.from(allItems)..sort((a, b) => b.id.compareTo(a.id)));

          contentSliver = SliverPadding(
            padding: .all(4),
            sliver: SliverWaterfallFlow(
              gridDelegate: const SliverWaterfallFlowDelegateWithMaxCrossAxisExtent(
                maxCrossAxisExtent: 300,
                crossAxisSpacing: 4,
                mainAxisSpacing: 4,
              ),
              delegate: SliverChildBuilderDelegate((context, index) {
                final item = sortedItems[index];
                final reaction = coordinator?.getReaction(isSearching ? (item as SearchResult).reactionId : (item as Reaction).id);
                if (reaction == null) return const SizedBox.shrink();
                return _GifTile(reaction: reaction);
              }, childCount: sortedItems.length),
            ),
          );
        }

        // One layout to rule them all
        return CustomScrollView(
          slivers: [
            AdaptiveHeightSliverPersistentHeader(floating: true, pinned: true, needRepaint: false, child: MySearchBar().center()),
            contentSliver,
          ],
        );
      },
    );
  }
}

class _GifTile extends StatefulWidget {
  final Reaction reaction;
  const _GifTile({required this.reaction});

  @override
  State<_GifTile> createState() => _GifTileState();
}

class _GifTileState extends State<_GifTile> with TickerProviderStateMixin {
  late final FPopoverController _menuController;

  @override
  void initState() {
    super.initState();
    _menuController = FPopoverController(vsync: this, shown: false);
  }

  @override
  void dispose() {
    _menuController.dispose();
    super.dispose();
  }

  void _hideMenu() {
    _menuController.hide();
  }

  Reaction get reaction => widget.reaction;

  @override
  Widget build(BuildContext context) {
    final colors = context.theme.colors;
    final c = coordinator!;

    final thumbnailStaticAbs = reaction.thumbnailStatic != null ? c.resolvePath(reaction.thumbnailStatic!) : null;
    final thumbnailAnimatedAbs = reaction.thumbnailAnimated != null ? c.resolvePath(reaction.thumbnailAnimated!) : null;
    final localFileAbs = reaction.localFile != null ? c.resolvePath(reaction.localFile!) : null;

    return FContextMenu(
      control: .managed(controller: _menuController),
      menu: [
        .group(
          children: [
            .item(
              prefix: const Icon(FLucideIcons.link, size: 16),
              title: const Text('Copy URL'),
              onPress: () async {
                _hideMenu();
                await Clipboard.setData(ClipboardData(text: reaction.url));
                if (mounted) {
                  showFToast(context: context, title: const Text('Link copied'));
                }
              },
            ),
            if (localFileAbs != null)
              .item(
                prefix: const Icon(FLucideIcons.clipboardCopy, size: 16),
                title: const Text('Copy file path'),
                onPress: () async {
                  _hideMenu();
                  await Clipboard.setData(ClipboardData(text: localFileAbs));
                  if (mounted) {
                    showFToast(context: context, title: const Text('Path copied'));
                  }
                },
              ),
            // .item(
            //   prefix: const Icon(FLucideIcons.externalLink, size: 16),
            //   title: const Text('Open in browser'),
            //   onPress: () {
            //     // TODO: url_launcher
            //   },
            // ),
          ],
        ),
        .group(
          children: [
            .item(
              prefix: const Icon(FLucideIcons.refreshCw, size: 16),
              title: const Text('Re-embed'),
              onPress: () async {
                _hideMenu();
                await c.reEmbedImage(reaction.id);
                if (mounted) {
                  showFToast(context: context, title: const Text('Re-embedded'));
                }
              },
            ),
            .item(
              prefix: const Icon(FLucideIcons.refreshCcw, size: 16),
              title: const Text('Re-generate thumbnail'),
              onPress: () async {
                _hideMenu();
                await c.regenerateThumbnails(reaction.id);
                refreshReactions();
                if (mounted) {
                  showFToast(context: context, title: const Text('Thumbnail regenerated'));
                }
              },
            ),
            if (!reaction.pinned)
              .item(
                prefix: const Icon(FLucideIcons.pin, size: 16),
                title: const Text('Pin locally'),
                onPress: () async {
                  _hideMenu();
                  await c.pinReaction(reaction.id);
                  refreshReactions();
                },
              ),
          ],
        ),
        .group(
          children: [
            .item(
              prefix: const Icon(FLucideIcons.trash2, size: 16),
              title: const Text('Delete'),
              onPress: () async {
                _hideMenu();
                await c.deleteReaction(reaction.id);
                refreshReactions();
                if (mounted) {
                  showFToast(context: context, title: const Text('Deleted'));
                }
              },
            ),
          ],
        ),
      ],
      child: FTappable(
        onPress: () async {
          await Clipboard.setData(ClipboardData(text: reaction.url));
          await incrementUsage(reaction.id);
          if (context.mounted) {
            showFToast(context: context, title: const Text('Link copied'));
          }
        },
        child: FCard.raw(
          clipBehavior: Clip.antiAlias,
          child: Stack(
            children: [
              SignalBuilder(
                builder: (context) {
                  final capturedDims = reactionDimensions.value[reaction.id];
                  final storedDims = (reaction.width != null && reaction.height != null && reaction.height! > 0)
                      ? (reaction.width!, reaction.height!)
                      : null;
                  final effectiveDims = capturedDims;
                  final aspectRatio = effectiveDims != null && effectiveDims.$2 > 0
                      ? effectiveDims.$1 / effectiveDims.$2
                      : (storedDims != null && storedDims.$2 > 0 ? storedDims.$1 / storedDims.$2 : 1.0);

                  return AspectRatio(
                    aspectRatio: aspectRatio,
                    child: Image(
                      image: ReactionImageProvider(
                        reactionId: reaction.id,
                        url: reaction.url,
                        thumbnailStaticPath: thumbnailStaticAbs,
                        thumbnailAnimatedPath: thumbnailAnimatedAbs,
                        localFilePath: localFileAbs,
                        animatedPreviewsEnabled: coordinator?.config.animatedPreviewsEnabled ?? false,
                        onDimensions: (w, h) {
                          final current = reactionDimensions.value[reaction.id];
                          if (current == null) {
                            final newDims = Map<String, (int, int)>.from(reactionDimensions.value);
                            newDims[reaction.id] = (w, h);
                            reactionDimensions.value = newDims;
                            coordinator?.updateDimensions(reaction.id, w, h);
                          }
                        },
                      ),
                      fit: BoxFit.cover,
                      errorBuilder: (context, error, stack) => Container(
                        color: colors.muted,
                        child: Icon(FLucideIcons.imageOff, size: 28, color: colors.mutedForeground),
                      ),
                      loadingBuilder: (context, child, loadingProgress) {
                        if (loadingProgress == null) return child;
                        return Container(
                          color: colors.muted,
                          child: Center(
                            child: SizedBox(
                              width: 24,
                              height: 24,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                value: loadingProgress.expectedTotalBytes != null
                                    ? loadingProgress.cumulativeBytesLoaded / loadingProgress.expectedTotalBytes!
                                    : null,
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  );
                },
              ),
              SignalBuilder(
                builder: (context) {
                  final status = ingestStatus.value[reaction.id];
                  if (status == null || status.isEmpty) {
                    return const SizedBox.shrink();
                  }
                  return Positioned(
                    left: 0,
                    right: 0,
                    bottom: 0,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [Colors.transparent, Colors.black.withValues(alpha: 0.7)],
                        ),
                      ),
                      child: Text(
                        status,
                        style: const TextStyle(color: Colors.white, fontSize: 11, decoration: null),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  );
                },
              ),
              if (reaction.usageCount > 0)
                Positioned(
                  right: 4,
                  top: 4,
                  child: FBadge(variant: FBadgeVariant.secondary, child: Text('${reaction.usageCount}')),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
