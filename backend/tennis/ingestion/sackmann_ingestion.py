"""
Sackmann Tennis Data Ingestion
-------------------------------
Loads ATP match data from local CSV files into tennis_players and tennis_matches tables.
Files must be copied into the container at /tmp/tennis_kaggle/tennis_atp/ before running.
"""

import csv
import io
import logging
import os
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

logger = logging.getLogger(__name__)

TENNIS_DATA_DIR = "/app/tennis_kaggle/tennis_atp"


class SackmannIngestion:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _fetch_csv(self, year: int) -> str:
        path = os.path.join(TENNIS_DATA_DIR, f"atp_matches_{year}.csv")
        logger.info(f"Reading {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_int(self, value: str):
        if value is None or value.strip() == "":
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _parse_date(self, value: str):
        if not value or len(value) != 8:
            return None
        try:
            return datetime.strptime(value, "%Y%m%d").date()
        except ValueError:
            return None

    async def _get_or_create_player(self, sackmann_id: int, name: str, hand: str, height: str, country: str) -> int:
        result = await self.session.execute(
            text("SELECT id FROM tennis_players WHERE sackmann_id = :sid"),
            {"sid": sackmann_id},
        )
        row = result.fetchone()
        if row:
            return row[0]

        result = await self.session.execute(
            text("""
                INSERT INTO tennis_players (sackmann_id, name, hand, height_cm, country)
                VALUES (:sid, :name, :hand, :height, :country)
                RETURNING id
            """),
            {
                "sid": sackmann_id,
                "name": name,
                "hand": hand,
                "height": self._parse_int(height),
                "country": country,
            },
        )
        new_id = result.fetchone()[0]
        return new_id

    async def ingest_season(self, year: int) -> int:
        csv_text = await self._fetch_csv(year)
        reader = csv.DictReader(io.StringIO(csv_text))

        inserted = 0
        skipped_duplicates = 0

        for row in reader:
            winner_sackmann_id = self._parse_int(row.get("winner_id"))
            loser_sackmann_id = self._parse_int(row.get("loser_id"))
            tourney_date = self._parse_date(row.get("tourney_date"))

            if not winner_sackmann_id or not loser_sackmann_id or not tourney_date:
                continue

            existing = await self.session.execute(
                text("SELECT id FROM tennis_matches WHERE tourney_id = :tid AND match_num = :mnum"),
                {"tid": row.get("tourney_id"), "mnum": self._parse_int(row.get("match_num"))},
            )
            if existing.fetchone():
                skipped_duplicates += 1
                continue

            winner_db_id = await self._get_or_create_player(
                winner_sackmann_id, row.get("winner_name"), row.get("winner_hand"),
                row.get("winner_ht"), row.get("winner_ioc"),
            )
            loser_db_id = await self._get_or_create_player(
                loser_sackmann_id, row.get("loser_name"), row.get("loser_hand"),
                row.get("loser_ht"), row.get("loser_ioc"),
            )

            await self.session.execute(
                text("""
                    INSERT INTO tennis_matches (
                        tourney_id, tourney_name, surface, draw_size, tourney_level,
                        tourney_date, match_num, best_of, round, score, minutes,
                        winner_id, winner_rank, winner_rank_points,
                        loser_id, loser_rank, loser_rank_points,
                        w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
                        l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced
                    ) VALUES (
                        :tourney_id, :tourney_name, :surface, :draw_size, :tourney_level,
                        :tourney_date, :match_num, :best_of, :round, :score, :minutes,
                        :winner_id, :winner_rank, :winner_rank_points,
                        :loser_id, :loser_rank, :loser_rank_points,
                        :w_ace, :w_df, :w_svpt, :w_1stIn, :w_1stWon, :w_2ndWon, :w_SvGms, :w_bpSaved, :w_bpFaced,
                        :l_ace, :l_df, :l_svpt, :l_1stIn, :l_1stWon, :l_2ndWon, :l_SvGms, :l_bpSaved, :l_bpFaced
                    )
                """),
                {
                    "tourney_id": row.get("tourney_id"),
                    "tourney_name": row.get("tourney_name"),
                    "surface": row.get("surface"),
                    "draw_size": self._parse_int(row.get("draw_size")),
                    "tourney_level": row.get("tourney_level"),
                    "tourney_date": tourney_date,
                    "match_num": self._parse_int(row.get("match_num")),
                    "best_of": self._parse_int(row.get("best_of")),
                    "round": row.get("round"),
                    "score": row.get("score"),
                    "minutes": self._parse_int(row.get("minutes")),
                    "winner_id": winner_db_id,
                    "winner_rank": self._parse_int(row.get("winner_rank")),
                    "winner_rank_points": self._parse_int(row.get("winner_rank_points")),
                    "loser_id": loser_db_id,
                    "loser_rank": self._parse_int(row.get("loser_rank")),
                    "loser_rank_points": self._parse_int(row.get("loser_rank_points")),
                    "w_ace": self._parse_int(row.get("w_ace")),
                    "w_df": self._parse_int(row.get("w_df")),
                    "w_svpt": self._parse_int(row.get("w_svpt")),
                    "w_1stIn": self._parse_int(row.get("w_1stIn")),
                    "w_1stWon": self._parse_int(row.get("w_1stWon")),
                    "w_2ndWon": self._parse_int(row.get("w_2ndWon")),
                    "w_SvGms": self._parse_int(row.get("w_SvGms")),
                    "w_bpSaved": self._parse_int(row.get("w_bpSaved")),
                    "w_bpFaced": self._parse_int(row.get("w_bpFaced")),
                    "l_ace": self._parse_int(row.get("l_ace")),
                    "l_df": self._parse_int(row.get("l_df")),
                    "l_svpt": self._parse_int(row.get("l_svpt")),
                    "l_1stIn": self._parse_int(row.get("l_1stIn")),
                    "l_1stWon": self._parse_int(row.get("l_1stWon")),
                    "l_2ndWon": self._parse_int(row.get("l_2ndWon")),
                    "l_SvGms": self._parse_int(row.get("l_SvGms")),
                    "l_bpSaved": self._parse_int(row.get("l_bpSaved")),
                    "l_bpFaced": self._parse_int(row.get("l_bpFaced")),
                },
            )
            inserted += 1

        await self.session.commit()
        logger.info(f"Season {year}: inserted {inserted}, skipped {skipped_duplicates} duplicates")
        return inserted