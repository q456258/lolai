class Champion:
    
    def formatChampion(self, champion: dict, version: str) -> dict:
        records = [
            self.formatChampionOverview(champion, version),
        ]
        for record in self.formatChampionSpell(champion, version):
            records.append(record)

        return records

    def formatChampionOverview(self, champion, version):
        id = champion["id"]
        key = champion["key"]
        title = champion["title"]
        partype = champion["partype"]
        allyTips = champion.get("allyTips", [])
        enemytips = champion.get("enemytips", [])
        tags = champion.get("tags", [])
        info = champion.get("info", {})
        stats = champion.get("stats", {})

        info_value = ", ".join(f'{k}:{v}' for k,v in info.items())
        stats_value = ", ".join(f'{k}:{v}' for k,v in stats.items())

        text = "\n".join(
            [
                f"id: {id}",
                f"key: {key}",
                f"name: {champion["name"]}",
                f"title: {title}",
                f"lore: {champion["lore"]}",
                f"blurb: {champion["blurb"]}",
                f"allyTips: {','.join(allyTips)}",
                f"enemytips: {','.join(enemytips)}",
                f"tags: {','.join(tags)}",
                f"partype: {partype}",
                f"info: {info_value}",
                f"stats: {stats_value}",
            ]
        )

        document_id = f"{key}:overview"
        return {
            "point_id": document_id,
            "title": title,
            "text": text,
            "metadata": {
                "entity_name": "champion",
                "document_id": document_id,
                "name": id,
                "tags": tags,
                "partype": partype,
                "info": info,
            }
        }
    
    def formatChampionSpell(self, champion, version):
        id = champion["id"]
        key = champion["key"]
        title = champion["title"]
        partype = champion["partype"]
        tags = champion.get("tags", [])
        info = champion.get("info", {})
        spells = champion.get("spells", [])
        passive = champion.get("passive", {})

        spell_value = [
            (
                f"parentId={id}, "
                f"championTitle={title}, "
                f"id={spell['id']}, "
                f"name={spell['name']}, "
                f"description={spell['description']}"
            )
            for spell in spells
        ]

        passive_value = (
            f"parentId={id}, "
            f"championTitle={title}, "
            f"name={passive['name']}, "
            f"description={passive['description']}"
        )
        spell_value.append(passive_value)

        spellType = ["spellQ","spellW","spellE","spellR","passive"]

        records = []
        for type, text in zip(spellType, spell_value):
            records.append( {
                "point_id": f"{key}:{type}",
                "title": title,
                "text": text,
                "metadata": {
                    "entity_name": "champion",
                    "parent_id": f"{key}:overview",
                    "document_id": f"{key}:{type}",
                    "name": id,
                    "tags": tags,
                    "partype": partype,
                    "info": info,
                }
            })


        return records