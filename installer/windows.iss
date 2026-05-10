; WhisperApp Windows installer — Inno Setup script
; Build: iscc /DAppVersion=1.1.0 installer\windows.iss

#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif

[Setup]
AppName=WhisperApp
AppVersion={#AppVersion}
AppPublisher=James Davies
AppPublisherURL=https://github.com/jlldavies/whisperapp
AppSupportURL=https://github.com/jlldavies/whisperapp/issues
AppUpdatesURL=https://github.com/jlldavies/whisperapp/releases
DefaultDirName={autopf}\WhisperApp
DefaultGroupName=WhisperApp
AllowNoIcons=yes
OutputDir=Output
OutputBaseFilename=WhisperApp-{#AppVersion}-setup
SetupIconFile=..\assets\icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest        ; install per-user, no UAC prompt needed
ArchitecturesInstallIn64BitMode=x64

; Minimum Windows 10
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupicon"; Description: "Start WhisperApp when I log in"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
; PyInstaller output — the full dist/WhisperApp directory
Source: "..\dist\WhisperApp\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\WhisperApp"; Filename: "{app}\WhisperApp.exe"
Name: "{group}\{cm:UninstallProgram,WhisperApp}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\WhisperApp"; Filename: "{app}\WhisperApp.exe"; Tasks: desktopicon
Name: "{userstartup}\WhisperApp"; Filename: "{app}\WhisperApp.exe"; Tasks: startupicon

[Run]
Filename: "{app}\WhisperApp.exe"; Description: "{cm:LaunchProgram,WhisperApp}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove ML packages installed at runtime into %APPDATA%\whisperapp-mlenv
; Comment this out if you want to keep downloaded models and packages
; Type: filesandordirs; Name: "{userappdata}\.whisperapp\mlenv"
