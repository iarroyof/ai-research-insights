SHELL := /bin/bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

COMPOSE_FILES := -f docker-compose.yml
DC := docker compose $(COMPOSE_FILES)

.PHONY: up down pull build rebuild logs ps stop rm

up:
	$(DC) up -d

down:
	$(DC) down

pull:
	$(DC) pull

build:
	$(DC) build

rebuild:
	$(DC) build --no-cache

logs:
	$(DC) logs -f

ps:
	$(DC) ps

stop:
	$(DC) stop

rm:
	$(DC) down -v --remove-orphans

# Helpful one-off commands
health:
	curl -s http://localhost:8080/api/health || true

openie:
	curl -s http://localhost:9006/health || true
