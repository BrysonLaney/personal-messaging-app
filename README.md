# Overview

This project is a small networking application that provides real-time group chat and file sharing, similar to a simplified self-hosted Discord.

The system has two main parts:

- A Python backend server (FastAPI) that exposes HTTP and WebSocket endpoints.
- A browser-based client (single `index.html`) that users open to chat and share files.

Basic usage:

1. Start the backend on your server:

   uvicorn main:app --host <server-ip> --port 8000

2. Open the frontend in a browser at:

   http://<server-ip>:8000

3. Register a user, log in, select or create a channel, then send messages and upload/download files.

The main goal of this project is to practice building a full client/server system: HTTP APIs, WebSockets, authentication, persistent storage, and a simple UI, all working together in a realistic networked application.

[Software Demo Video](https://youtu.be/JawJpUI2oyU)

---

# Network Communication

- **Architecture:** Client/Server  
  - Backend: FastAPI app running on a Linux server.  
  - Client: Single-page web app running in the browser.

- **Transport:**  
  - TCP  
  - Backend listens on port `8000`.

- **Protocols and endpoints:**
  - HTTP (REST-style) for:
    - `POST /auth/register` – user registration  
    - `POST /auth/login` – user login  
    - `GET /me` – current user info  
    - `GET /channels` – list user channels  
    - `POST /channels` – create channel  
    - `POST /channels/{id}/join` – join existing channel  
    - `GET /channels/{id}/messages` – list recent messages  
    - `POST /channels/{id}/messages` – create message  
    - `POST /channels/{id}/files` – upload file  
    - `GET /files/{id}` – download file
  - WebSockets for real-time chat:
    - `ws://<server-ip>:8000/ws/channels/{channel_id}?token=<jwt>`

- **Authentication:**
  - JWT bearer tokens.
  - HTTP requests include `Authorization: Bearer <token>`.
  - WebSocket connections pass the same token as a query parameter.

---

# Development Environment

- **OS / Tools**
  - Ubuntu Server
  - Python 3.12 with `venv`
  - Uvicorn (ASGI server)
  - VS Code and terminal

- **Backend**
  - Language: Python
  - Frameworks and libraries:
    - FastAPI
    - Uvicorn
    - SQLAlchemy
    - python-jose
    - passlib
    - python-multipart

- **Frontend**
  - Plain HTML, CSS, and JavaScript (no external JS framework)
  - Browser `fetch` API for HTTP
  - Browser `WebSocket` API for real-time messaging

---

# Useful Websites

* https://fastapi.tiangolo.com/
* https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
* https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API
* https://docs.sqlalchemy.org/
* https://passlib.readthedocs.io/

---

# Future Work

* Add direct messages (DMs) and a friend list system.
* Show online/offline presence and a member list per channel.
* Add HTTPS support with a reverse proxy (nginx or Caddy).
* Improve error handling and logging.
* Add channel permissions and moderation roles.
