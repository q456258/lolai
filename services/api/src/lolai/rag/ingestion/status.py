class Status:
    def formatStatus(self, record: dict) -> dict:
        name = record.get("name", [])
        parent = record.get("parent", [])
        desc = record.get("description",{})

        text = "\n".join(
            [
                f"name: {name}",
                f"type: status",
                f"description: belong to category {parent}, {desc}",
            ]
        )

        return {
            "point_id": name,
            "title": record["name"],
            "text": text,
            "metadata": {
                "entity_name": "status",
            }
        }