#!/usr/bin/env bash
set -e

echo "🏈 Football Predictor — Day 1 Setup"
echo "======================================"

# 1. Copy env file if not present
if [ ! -f .env ]; then
  cp .env.example .env
  echo "✅ Created .env from .env.example — add your API keys before Day 2"
else
  echo "ℹ️  .env already exists"
fi

# 2. Start Postgres + Redis only first
echo ""
echo "🐳 Starting PostgreSQL and Redis..."
docker compose -f docker/docker-compose.yml up -d postgres redis

# 3. Wait for Postgres to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker exec football_postgres pg_isready -U football -d football_db > /dev/null 2>&1; do
  sleep 1
done
echo "✅ PostgreSQL is ready"

# 4. Run Alembic migrations
echo ""
echo "🗄️  Running database migrations..."
cd backend
pip install -q alembic psycopg2-binary pydantic-settings python-dotenv sqlalchemy
alembic upgrade head
cd ..
echo "✅ Migrations complete — all tables created"

# 5. Start the full stack
echo ""
echo "🚀 Starting full stack (API + Frontend)..."
docker compose -f docker/docker-compose.yml up -d

echo ""
echo "======================================"
echo "✅ Day 1 Complete!"
echo ""
echo "  API:      http://localhost:8000"
echo "  Docs:     http://localhost:8000/docs"
echo "  Health:   http://localhost:8000/api/v1/health"
echo "  Frontend: http://localhost:3000  (after npm install, ~1 min)"
echo ""
echo "📋 Next: Add your API keys to .env, then confirm for Day 2"
echo "======================================"
