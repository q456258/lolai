class Item:
    
    def formatItem(self, item_id: str, item: dict, version: str) -> dict:
        into = item.get("into", [])
        tags = item.get("tags", [])
        gold = item.get("gold",{})
        maps = item.get("maps", {})
        stats = item.get("stats", {})

        gold_value = ', '.join(f'{k}:{v}' for k,v in gold.items())
        maps_value = ', '.join(f'{k}:{v}' for k,v in maps.items())
        stats_value = ', '.join(f'{k}:{v}' for k,v in stats.items())
        text = "\n".join(
            [
                f"name: {item["name"]}",
                f"type: item",
                f"description: {item["description"]}",
                f"plaintext: {item["plaintext"]}",
                f"into: {', '.join(into)}"
                f"gold: {gold_value}",
                f"tags: {', '.join(tags)}"
                f"maps: {maps_value}",
                f"stats: {stats_value}",
            ]
        )

        return {
            "point_id": item_id,
            "title": item["name"],
            "text": text,
            "metadata": {
                "entity_name": "item",
                "version": version,
                "item_id": item_id,
                "into": into,
                "tags": tags,
                "gold": gold,
                "maps": maps,
                "stats": stats,
            }
        }