# Installation Guide (Windows)

This guide explains how to install **Git**, clone the repository, and run the project on Windows.

---

## 1. Install Git (Only once)

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

### Git Installation Screenshots

<img src="screenshots/git_install.png" width="400">
<img src="screenshots/git_install2.png" width="400">
<img src="screenshots/git_install3.png" width="400">
<img src="screenshots/git_install4.png" width="400">
<img src="screenshots/git_install5.png" width="400">
<img src="screenshots/git_install6.png" width="400">
<img src="screenshots/git_install7.png" width="400">
<img src="screenshots/git_install8.png" width="400">
<img src="screenshots/git_install9.png" width="400">
<img src="screenshots/git_install10.png" width="400">
<img src="screenshots/git_install11.png" width="400">
<img src="screenshots/git_install12.png" width="400">
<img src="screenshots/git_install13.png" width="400">
<img src="screenshots/git_install14.png" width="400">
<img src="screenshots/git_install15.png" width="400">
<img src="screenshots/git_install16.png" width="400">
<img src="screenshots/git_install17.png" width="400">

### How to Open PowerShell

1. Click the **Start** button (or press the **Windows key**)
2. Type **PowerShell**
3. Click **Windows PowerShell**

![Open PowerShell](screenshots/powershell.png)

## 2. Clone the Repository (Only once)

### Choose a folder where you want the project to live

Open **PowerShell** and run:
```
git clone https://github.com/jsfakian/dicom_anonymization.git
cd dicom_anonymization
```

![Clone the repository](screenshots/clone_repo2.png)

## 3. Run the Application (Many times)

Open dicom_anonymization folder and double-click on DICOM-DeID.exe

