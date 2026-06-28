"""
Tennis-Data.co.uk Odds Ingestion
----------------------------------
Loads bookmaker odds + exact total-games (derived from set scores) and
matches each row to an existing tennis_matches row by date + player names.

Name matching is fuzzy because tennis-data.co.uk uses "Lastname F." format
while our Sackmann data uses "Firstname Lastname" — exact string match
won't work, so we match on (date, last name substring).
"""

import logging
from datetime import timedelta
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)


def compute_total_games(row) -> int | None:
    total = 0
    found_any = False
    for i in range(1, 6):
        w = row.get(f"W{i}")
        l = row.get(f"L{i}")
        if pd.notna(w) and pd.notna(l):
            total += int(w) + int(l)
            found_any = True
    return total if found_any else None


class TennisOddsIngestion:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def ingest_file(self, xlsx_path: str) -> dict:
        df = pd.read_excel(xlsx_path)
        inserted, matched, skipped = 0, 0, 0

        for _, row in df.iterrows():
            match_date = row.get("Date")
            if pd.isna(match_date):
                skipped += 1
                continue
            match_date = match_date.date()

            winner_name = str(row.get("Winner", "")).strip()
            loser_name = str(row.get("Loser", "")).strip()
            if not winner_name or not loser_name:
                skipped += 1
                continue

            total_games = compute_total_games(row)

            # Try to find the matching real match by date + last-name fragment
            winner_last = winner_name.split()[0].replace("-", " ").strip() if winner_name else ""
            loser_last = loser_name.split()[0].replace("-", " ").strip() if loser_name else ""

            matched_row = await self.session.execute(
                text("""
                    SELECT tm.id FROM tennis_matches tm
                    JOIN tennis_players w ON tm.winner_id = w.id
                    JOIN tennis_players l ON tm.loser_id = l.id
                    WHERE tm.tourney_date BETWEEN :date_start AND :date_end
                      AND tm.surface = :surface
                      AND w.name ILIKE :winner_pattern
                      AND l.name ILIKE :loser_pattern
                    LIMIT 1
                """),
                {
                    "date_start": match_date - timedelta(days=21),
                    "date_end": match_date + timedelta(days=3),
                    "surface": row.get("Surface"),
                    "winner_pattern": f"%{winner_last}%",
                    "loser_pattern": f"%{loser_last}%",
                },
            )
            matched_id = matched_row.scalar()
            if matched_id:
                matched += 1

            try:
                await self.session.execute(
                    text("""
                        INSERT INTO tennis_odds (
                            match_date, tournament, surface, winner_name, loser_name,
                            winner_rank, loser_rank, b365_winner_odds, b365_loser_odds,
                            avg_winner_odds, avg_loser_odds, actual_total_games,
                            matched_tennis_match_id
                        ) VALUES (
                            :match_date, :tournament, :surface, :winner_name, :loser_name,
                            :winner_rank, :loser_rank, :b365w, :b365l, :avgw, :avgl,
                            :total_games, :matched_id
                        )
                        ON CONFLICT (match_date, winner_name, loser_name) DO NOTHING
                    """),
                    {
                        "match_date": match_date,
                        "tournament": row.get("Tournament"),
                        "surface": row.get("Surface"),
                        "winner_name": winner_name,
                        "loser_name": loser_name,
                        "winner_rank": int(row["WRank"]) if pd.notna(row.get("WRank")) else None,
                        "loser_rank": int(row["LRank"]) if pd.notna(row.get("LRank")) else None,
                        "b365w": row.get("B365W") if pd.notna(row.get("B365W")) else None,
                        "b365l": row.get("B365L") if pd.notna(row.get("B365L")) else None,
                        "avgw": row.get("AvgW") if pd.notna(row.get("AvgW")) else None,
                        "avgl": row.get("AvgL") if pd.notna(row.get("AvgL")) else None,
                        "total_games": total_games,
                        "matched_id": matched_id,
                    },
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Skipped row: {e}")
                skipped += 1

        await self.session.commit()
        return {"inserted": inserted, "matched": matched, "skipped": skipped, "total_rows": len(df)}