# Whisper STT Web UI — статический nginx (самодостаточный index.html, без сборки).
# Отдельно от API-образа: собирается быстро и независимо.
FROM nginx:1.27-alpine

COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
