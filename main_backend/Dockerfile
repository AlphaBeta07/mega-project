# StudySnap AI - Main Application Gateway (Node.js/Express) Dockerfile
FROM node:20-alpine

WORKDIR /app

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy package descriptors
COPY package*.json ./

# Install production dependencies
RUN npm ci --only=production

# Copy application files
COPY . .

# Ensure persistent data folder exists
RUN mkdir -p data uploads

# Expose Node.js gateway port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Start express server
CMD ["node", "server.js"]
