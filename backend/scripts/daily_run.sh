#!/bin/bash
# MRXD3000 Predict - Daily automation
echo Starting daily run...
cd /app
python3 -c "import asyncio,sys; sys.path.insert(0,'/app'); exec(open('/app/scripts/daily_auto.py').read())"
echo Done.
