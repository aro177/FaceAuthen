# Python Project Setup (Windows & Linux)

This guide explains how to set up and run this project on both **Windows (PowerShell)** and **Linux (Ubuntu/Debian-based)** systems.

---

# Windows Setup (PowerShell)

## Prerequisites

Make sure Python 3 is installed.

### Install via Winget

```powershell
winget install Python.Python.3
```

Or download manually: https://www.python.org/downloads/

> Remember to check **"Add Python to PATH"**

---

## Steps

### 1. Create virtual environment

```powershell
python -m venv venv
```

### 2. Activate virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

#### If execution policy error occurs:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# Linux Setup (Ubuntu/Debian)

## Install Python & tools

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

---

## Steps

### 1. Create virtual environment

```bash
python3 -m venv venv
```

### 2. Activate virtual environment

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---
