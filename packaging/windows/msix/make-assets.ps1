# Generates the placeholder Store logo assets referenced by AppxManifest.xml.
# Replace these with real artwork before a polished Store listing. Run: pwsh make-assets.ps1
Add-Type -AssemblyName System.Drawing
$dir = Join-Path $PSScriptRoot 'assets'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$bg = [System.Drawing.ColorTranslator]::FromHtml('#1b1f2a')
$fg = [System.Drawing.ColorTranslator]::FromHtml('#8fb3ff')

function New-Logo([int]$size, [string]$name) {
    $bmp = New-Object System.Drawing.Bitmap($size, $size)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'
    $g.Clear($bg)
    # A simple ">" proxy glyph, scaled to the icon.
    $pen = New-Object System.Drawing.Pen($fg, [math]::Max(2, $size / 12))
    $pen.StartCap = 'Round'; $pen.EndCap = 'Round'; $pen.LineJoin = 'Round'
    $a = $size * 0.34; $b = $size * 0.5; $c = $size * 0.66
    $pts = [System.Drawing.PointF[]]@(
        (New-Object System.Drawing.PointF($a, $a)),
        (New-Object System.Drawing.PointF($c, $b)),
        (New-Object System.Drawing.PointF($a, $c))
    )
    $g.DrawLines($pen, $pts)
    $g.Dispose()
    $bmp.Save((Join-Path $dir $name), [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Write-Output "wrote $name ($size x $size)"
}

New-Logo 44  'Square44x44Logo.png'
New-Logo 150 'Square150x150Logo.png'
New-Logo 50  'StoreLogo.png'
