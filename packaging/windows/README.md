# Windows installers

Two Windows packagings of the self-contained `ai-proxy.exe` (PyInstaller binary — no
Python required). Both are built from the frozen binary in `dist/bin/ai-proxy.exe`.

## MSI (direct download)

A classic installer: installs into `Program Files\AI Proxy`, adds a Start Menu shortcut,
and registers a normal uninstaller. **Installs unsigned** (SmartScreen shows an
"unknown publisher" prompt you click through); sign the `.msi` with a code-signing
certificate to remove that.

```powershell
# needs WiX v5:  dotnet tool install --global wix
wix build packaging/windows/msi/ai-proxy.wxs -bindpath dist/bin -d Version=0.1.0 -o dist/ai-proxy-0.1.0.msi
```

Attached to each GitHub Release. Install by double-clicking; then launch **AI Proxy** from
the Start Menu (UI at <http://127.0.0.1:8000/__proxy/>).

## MSIX (Microsoft Store)

A full-trust desktop package for Store distribution (the Store gives you auto-update and
free signing). The frozen exe runs unsandboxed via `Windows.FullTrustApplication` +
`runFullTrust`, so the local proxy has no AppContainer loopback restriction.

```powershell
pwsh packaging/windows/msix/build-msix.ps1 -Version 0.1.0 -BinDir dist/bin -OutFile dist/ai-proxy-0.1.0.msix
# local install testing only (sign + trust a throwaway cert):
pwsh packaging/windows/msix/build-msix.ps1 -Version 0.1.0 -SelfSign
```

### Submitting to the Store

MSIX **cannot be installed unless signed** — but for the Store you upload it *unsigned*
and the Store signs it. Steps:

1. Create a **Partner Center** account (one-time $19 for an individual) and **reserve the
   app name**.
2. On the app's **Product Identity** page, copy the assigned **Package/Identity Name**,
   **Publisher** (`CN=...`), and **Publisher display name**.
3. Put those three values into `packaging/windows/msix/AppxManifest.xml` (they must match
   exactly, or the Store rejects the upload).
4. Build the `.msix` (command above) and **upload it** in a new submission.
5. Replace the placeholder logos in `packaging/windows/msix/assets/` with real artwork for
   a polished listing (regenerate placeholders with `make-assets.ps1`).

The release CI produces the `.msix` as a build artifact (not a release asset, since an
unsigned MSIX isn't directly installable) — download it from the workflow run to submit.
