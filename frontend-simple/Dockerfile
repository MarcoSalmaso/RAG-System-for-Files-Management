FROM nginx:alpine

# Copy HTML file to nginx directory
COPY index.html /usr/share/nginx/html/

# Create proper nginx config
RUN cat > /etc/nginx/conf.d/default.conf <<EOF
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;
    
    location / {
        try_files \$uri \$uri/ /index.html;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
}
EOF

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]