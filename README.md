# Python Project Setup (Windows)

This guide explains how to set up and run this project using **Windows PowerShell**.

---

## Prerequisites

Make sure you have **Python 3** installed on your system.

### Option 1: Install via Winget

```powershell
winget install Python.Python.3
```

### Option 2: Manual Installation

Download and install Python from: https://www.python.org/downloads/

> Ensure you check **"Add Python to PATH"** during installation.

---

## ⚙️ Setup Instructions

### 1. Create a Virtual Environment

```powershell
python -m venv venv
```

---

### 2. Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

#### If you encounter execution policy error:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

