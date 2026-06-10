
import asyncio, sys, os
sys.path.insert(0, '/app')

async def run():
    os.environ.setdefault('TELEGRAM_BOT_TOKEN', '8853528488:AAHfzkZ8sX9wAL6vHUNYladT6-TPnX5eciQ')
    os.environ.setdefault('TELEGRAM_CHAT_ID', '5996553176')
    from app.database import AsyncSessionLocal
    from app.services.prediction_service import prediction_service
    from app.services.alert_engine import run_alert_engine
    from sqlalchemy import text

    async with AsyncSessionLocal() as db:
        r = await db.execute(text("SELECT api_id FROM matches WHERE status='NS' ORDER BY match_date LIMIT 20"))
        ids = [row[0] for row in r.fetchall()]
        print(f'Predicting {len(ids)} upcoming matches...')
        for api_id in ids:
            try:
                await prediction_service.predict_for_match(api_id, db)
                print(f'  Predicted {api_id}')
            except Exception as e:
                print(f'  Failed {api_id}: {e}')

    async with AsyncSessionLocal() as db:
        result = await run_alert_engine(db)
        print(f'Alerts: {result}')

asyncio.run(run())
