import asyncio, sys, os
sys.path.insert(0, '/app')

async def run():
    from app.database import AsyncSessionLocal
    from app.services.prediction_service import prediction_service
    from app.services.alert_engine import run_alert_engine
    from sqlalchemy import text

    if not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('TELEGRAM_CHAT_ID'):
        print('WARNING: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in this '
              'environment — alerts will run but Telegram messages will be skipped.')

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
