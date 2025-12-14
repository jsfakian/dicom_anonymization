# Installation Guide (Windows)

This guide explains how to install **Git**, clone the repository, install **Python**, and run the project using a **virtual environment** on Windows.

---

## 1. Install Git

Git is required to download (clone) the project from GitHub.

### Steps
1. Go to: https://git-scm.com/download/win
2. Download **Git for Windows**
3. Run the installer
4. During installation:
   - Keep the **default options**
   - Make sure **"Git from the command line and also from 3rd-party software"** is selected
5. Finish installation

### Verify Git installation
Open **PowerShell** and run: `git --version`

You should see something like: `git version 2.x.x`

### How to Open PowerShell

1. Click the **Start** button (or press the **Windows key**)
2. Type **PowerShell**
3. Click **Windows PowerShell**

## 2. Clone the Repository

### Choose a folder where you want the project to live

Open **PowerShell** and run:
```
git clone https://github.com/jsfakian/dicom_anonymization.git
cd dicom_anonymization
```

## 3. Run the Application

Open **PowerShell** and run: `python anony.py`

