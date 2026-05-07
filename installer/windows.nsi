; WhisperApp Windows Installer — NSIS script
; Prerequisites: NSIS 3.x  (https://nsis.sourceforge.io)
; Run after: pyinstaller whisperapp.spec
; Usage: makensis installer\windows.nsi

!define APP_NAME      "WhisperApp"
!define APP_VERSION   "1.1.0"
!define APP_PUBLISHER "WhisperApp"
!define APP_URL       "http://127.0.0.1:7860"
!define INSTALL_DIR   "$PROGRAMFILES64\${APP_NAME}"
!define UNINSTALL_REG "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

Name "${APP_NAME} ${APP_VERSION}"
OutFile "WhisperApp-${APP_VERSION}-Setup.exe"
InstallDir "${INSTALL_DIR}"
InstallDirRegKey HKLM "${UNINSTALL_REG}" "InstallLocation"
RequestExecutionLevel admin
SetCompressor /SOLID lzma

; Modern UI
!include "MUI2.nsh"
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

; -----------------------------------------------------------------------
; Install
; -----------------------------------------------------------------------
Section "Install"
    SetOutPath "${INSTALL_DIR}"
    File /r "..\dist\WhisperApp\*.*"

    ; Start menu shortcut
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" \
        "${INSTALL_DIR}\WhisperApp.exe"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" \
        "${INSTALL_DIR}\Uninstall.exe"

    ; Desktop shortcut
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" \
        "${INSTALL_DIR}\WhisperApp.exe"

    ; Add to PATH so `whisperapp` CLI works from any terminal
    EnVar::SetHKLM
    EnVar::AddValue "PATH" "${INSTALL_DIR}"

    ; Register uninstaller
    WriteRegStr HKLM "${UNINSTALL_REG}" "DisplayName"     "${APP_NAME}"
    WriteRegStr HKLM "${UNINSTALL_REG}" "DisplayVersion"  "${APP_VERSION}"
    WriteRegStr HKLM "${UNINSTALL_REG}" "Publisher"       "${APP_PUBLISHER}"
    WriteRegStr HKLM "${UNINSTALL_REG}" "InstallLocation" "${INSTALL_DIR}"
    WriteRegStr HKLM "${UNINSTALL_REG}" "UninstallString" \
        '"${INSTALL_DIR}\Uninstall.exe"'
    WriteRegDWORD HKLM "${UNINSTALL_REG}" "NoModify" 1
    WriteRegDWORD HKLM "${UNINSTALL_REG}" "NoRepair"  1

    WriteUninstaller "${INSTALL_DIR}\Uninstall.exe"

    ; Launch on install
    Exec '"${INSTALL_DIR}\WhisperApp.exe"'
SectionEnd

; -----------------------------------------------------------------------
; Uninstall
; -----------------------------------------------------------------------
Section "Uninstall"
    ; Stop running instance
    ExecWait 'taskkill /F /IM WhisperApp.exe'

    ; Remove startup registry entry written by the app
    DeleteRegValue HKCU \
        "Software\Microsoft\Windows\CurrentVersion\Run" "${APP_NAME}"

    ; Remove files
    RMDir /r "${INSTALL_DIR}"

    ; Remove shortcuts
    Delete "$DESKTOP\${APP_NAME}.lnk"
    RMDir /r "$SMPROGRAMS\${APP_NAME}"

    ; Remove PATH entry
    EnVar::SetHKLM
    EnVar::DeleteValue "PATH" "${INSTALL_DIR}"

    ; Remove uninstall registry key
    DeleteRegKey HKLM "${UNINSTALL_REG}"
SectionEnd
