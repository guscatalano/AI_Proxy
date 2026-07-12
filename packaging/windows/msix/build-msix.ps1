# Packs the AI Proxy MSIX from the frozen binary. Produces an UNSIGNED .msix suitable
# for Microsoft Store submission (the Store signs it on upload). For local install
# testing, pass -SelfSign to sign it with a throwaway certificate whose subject matches
# the manifest Publisher (you must then trust that cert to install).
#
#   pwsh packaging/windows/msix/build-msix.ps1 -Version 0.1.0 -BinDir dist/bin -OutFile dist/ai-proxy-0.1.0.msix
param(
    [string]$Version = "0.1.0",
    [string]$BinDir  = "dist/bin",
    [string]$OutFile = "dist/ai-proxy.msix",
    [switch]$SelfSign
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot

$makeappx = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\makeappx.exe" |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $makeappx) { throw "makeappx.exe not found (install the Windows 10/11 SDK)" }

$stage = Join-Path ([System.IO.Path]::GetFullPath((Join-Path $here '..\..\..\build'))) 'msix-stage'
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Join-Path $stage 'assets') | Out-Null

# Stamp a 4-part version (x.y.z.0) into the manifest.
$four = (($Version -split '\.')[0..2] -join '.') + '.0'
(Get-Content (Join-Path $here 'AppxManifest.xml') -Raw) `
    -replace 'Version="0\.1\.0\.0"', "Version=`"$four`"" |
    Set-Content (Join-Path $stage 'AppxManifest.xml') -Encoding UTF8

Copy-Item (Join-Path $BinDir 'ai-proxy.exe') $stage
Copy-Item (Join-Path $here 'assets\*') (Join-Path $stage 'assets')

$OutFull = [System.IO.Path]::GetFullPath($OutFile)
New-Item -ItemType Directory -Force -Path (Split-Path $OutFull) | Out-Null
& $makeappx pack /d $stage /p $OutFull /o
if ($LASTEXITCODE -ne 0) { throw "makeappx failed ($LASTEXITCODE)" }
Write-Output "built $OutFull"

if ($SelfSign) {
    $signtool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe" |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
    # The signing cert subject must match the manifest Publisher exactly.
    $publisher = ([xml](Get-Content (Join-Path $here 'AppxManifest.xml'))).Package.Identity.Publisher
    $cert = New-SelfSignedCertificate -Type Custom -Subject $publisher `
        -KeyUsage DigitalSignature -FriendlyName "AI Proxy MSIX test" `
        -CertStoreLocation "Cert:\CurrentUser\My" `
        -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}")
    $pfx = Join-Path $stage 'test.pfx'
    $pw = ConvertTo-SecureString -String "test" -Force -AsPlainText
    Export-PfxCertificate -Cert "Cert:\CurrentUser\My\$($cert.Thumbprint)" -FilePath $pfx -Password $pw | Out-Null
    & $signtool sign /fd SHA256 /a /f $pfx /p "test" $OutFull
    Write-Output "self-signed with throwaway cert (thumbprint $($cert.Thumbprint)); install that cert to sideload."
}
