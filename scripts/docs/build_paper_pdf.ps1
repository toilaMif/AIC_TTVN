param(
    [string]$Workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
)

$ErrorActionPreference = 'Stop'
$paperPath = Join-Path $Workspace 'docs\paper\AIC_TTVN_paper.md'
$pdfPath = Join-Path $Workspace 'docs\paper\AIC_TTVN_paper.pdf'
$imageDir = Join-Path $Workspace 'docs\references\prak-v4'
$htmlPath = Join-Path ([System.IO.Path]::GetTempPath()) ("aic-ttvn-paper-{0}.html" -f ([guid]::NewGuid().ToString('N')))
if (-not (Test-Path -LiteralPath $imageDir)) { throw 'Could not locate docs/references/prak-v4.' }

function HtmlEncode([string]$text) {
    return [System.Net.WebUtility]::HtmlEncode($text)
}

function InlineMarkup([string]$text) {
    $encoded = HtmlEncode $text
    $encoded = [regex]::Replace($encoded, '\*\*(.+?)\*\*', '<strong>$1</strong>')
    $encoded = [regex]::Replace($encoded, '\*(.+?)\*', '<em>$1</em>')
    $encoded = [regex]::Replace($encoded, '`(.+?)`', '<code>$1</code>')
    $encoded = [regex]::Replace($encoded, '\\[(]|\\[)]|\\\[|\\\]', '')
    return $encoded
}

function FileUri([string]$path) {
    return ([uri](Resolve-Path -LiteralPath $path).Path).AbsoluteUri
}

$lines = Get-Content -LiteralPath $paperPath -Encoding UTF8
$body = New-Object System.Text.StringBuilder
$inCode = $false
$codeLines = New-Object System.Collections.Generic.List[string]
$inList = $false
$tableLines = New-Object System.Collections.Generic.List[string]

function Close-List {
    if ($script:inList) {
        [void]$script:body.AppendLine('</ul>')
        $script:inList = $false
    }
}

function Flush-Table {
    if ($script:tableLines.Count -lt 2) { return }
    [void]$script:body.AppendLine('<table>')
    for ($rowIndex = 0; $rowIndex -lt $script:tableLines.Count; $rowIndex++) {
        if ($rowIndex -eq 1 -and $script:tableLines[$rowIndex] -match '^\s*\|?[\s:\-|]+\|?\s*$') { continue }
        $cells = $script:tableLines[$rowIndex].Trim().Trim('|').Split('|')
        $tag = if ($rowIndex -eq 0) { 'th' } else { 'td' }
        [void]$script:body.AppendLine('<tr>')
        foreach ($cell in $cells) {
            [void]$script:body.AppendLine("<$tag>$(InlineMarkup $cell.Trim())</$tag>")
        }
        [void]$script:body.AppendLine('</tr>')
    }
    [void]$script:body.AppendLine('</table>')
    $script:tableLines.Clear()
}

foreach ($line in $lines) {
    if ($line -eq '```text') {
        Close-List; Flush-Table
        $inCode = $true
        $codeLines.Clear()
        continue
    }
    if ($line -eq '```' -and $inCode) {
        [void]$body.AppendLine('<pre>' + (HtmlEncode ($codeLines -join "`n")) + '</pre>')
        $inCode = $false
        continue
    }
    if ($inCode) { $codeLines.Add($line); continue }

    if ($line.Trim().StartsWith('|')) {
        Close-List
        $tableLines.Add($line)
        continue
    } elseif ($tableLines.Count -gt 0) {
        Flush-Table
    }

    if ($line -match '^# (.+)$') {
        Close-List
        [void]$body.AppendLine('<section class="title-page">')
        [void]$body.AppendLine('<div class="brand">AIC 2026 &middot; VIDEO RETRIEVAL</div>')
        [void]$body.AppendLine('<h1>' + (InlineMarkup $Matches[1]) + '</h1>')
        [void]$body.AppendLine('<div class="title-rule"></div>')
        continue
    }
    if ($line -match '^\*\*Anonymous authors\*\*') {
        [void]$body.AppendLine('<p class="authors">AIC-TTVN Development Team</p>')
        continue
    }
    if ($line -match '^\*\*Affiliation:') {
        [void]$body.AppendLine('<p class="affiliation">Vietnamese AI Challenge 2026</p>')
        continue
    }
    if ($line -match '^\*\*Corresponding author:') {
        [void]$body.AppendLine('<p class="date">System paper draft &middot; August 2026</p>')
        [void]$body.AppendLine('<div class="cover-flow"><span>Vietnamese Query</span><b>&rarr;</b><span>Multimodal Retrieval</span><b>&rarr;</b><span>Video Moment</span></div>')
        [void]$body.AppendLine('</section><div class="page-break"></div>')
        continue
    }
    if ($line -match '^## (.+)$') {
        Close-List
        $heading = $Matches[1]
        [void]$body.AppendLine('<h2>' + (InlineMarkup $heading) + '</h2>')
        if ($heading -eq '4. System Architecture') {
            [void]$body.AppendLine(@'
<figure class="architecture">
  <div class="arch-row"><div class="arch-box query">Vietnamese query<br><small>translation · entity · temporal parsing</small></div></div>
  <div class="arrow">&darr;</div>
  <div class="arch-row four"><div class="arch-box">Vision-language<br>CLIP / SigLIP</div><div class="arch-box">OCR<br>on-screen text</div><div class="arch-box">ASR<br>spoken content</div><div class="arch-box">Objects<br>labels & regions</div></div>
  <div class="arrow">&darr;</div>
  <div class="arch-row"><div class="arch-box fusion">Calibrated multimodal fusion &middot; temporal/spatial reranking &middot; diversification</div></div>
  <div class="arrow">&darr;</div>
  <div class="arch-row"><div class="arch-box output">Interactive keyframe grid · video playback · AIC frame export</div></div>
  <figcaption>Figure 1. Overall architecture of the proposed AIC‑TTVN retrieval system.</figcaption>
</figure>
'@)
        }
        if ($heading -eq '2. Related Work') {
            $src = FileUri (Join-Path $imageDir 'page-01.png')
            [void]$body.AppendLine("<figure><img src='$src'><figcaption>Figure 2. First page of PraK V4, the main system-paper reference supplied with this project.</figcaption></figure>")
        }
        if ($heading -eq '5. Interactive Interface and Submission Workflow') {
            $src = FileUri (Join-Path $imageDir 'page-03.png')
            [void]$body.AppendLine("<figure><img src='$src'><figcaption>Figure 3. PraK V4 result interface and localized-query discussion used as interaction-design inspiration.</figcaption></figure>")
        }
        continue
    }
    if ($line -match '^### (.+)$') {
        Close-List
        [void]$body.AppendLine('<h3>' + (InlineMarkup $Matches[1]) + '</h3>')
        continue
    }
    if ($line -match '^---$') {
        Close-List
        [void]$body.AppendLine('<hr>')
        continue
    }
    if ($line -match '^[-*] (.+)$') {
        if (-not $inList) { [void]$body.AppendLine('<ul>'); $inList = $true }
        [void]$body.AppendLine('<li>' + (InlineMarkup $Matches[1]) + '</li>')
        continue
    }
    if ($line -match '^\d+\. (.+)$') {
        if (-not $inList) { [void]$body.AppendLine('<ul class="numbered">'); $inList = $true }
        [void]$body.AppendLine('<li>' + (InlineMarkup $Matches[1]) + '</li>')
        continue
    }
    if ([string]::IsNullOrWhiteSpace($line)) { Close-List; continue }
    Close-List
    [void]$body.AppendLine('<p>' + (InlineMarkup $line) + '</p>')
}
Close-List
Flush-Table

$appendix = New-Object System.Text.StringBuilder
[void]$appendix.AppendLine('<div class="page-break"></div><h2>Appendix A. Supplied PraK V4 Reference Images</h2>')
foreach ($n in 1..5) {
    $src = FileUri (Join-Path $imageDir ("page-{0:D2}.png" -f $n))
    [void]$appendix.AppendLine("<figure class='source-page'><img src='$src'><figcaption>Appendix Figure A$n. Supplied page image $n of the PraK V4 reference.</figcaption></figure>")
}

$css = @'
@page { size: A4; margin: 18mm 17mm 20mm 17mm; }
* { box-sizing: border-box; }
body { font-family: "Times New Roman", serif; color: #172033; font-size: 10.5pt; line-height: 1.43; margin: 0; }
h1 { font-family: Arial, sans-serif; font-size: 30pt; line-height: 1.08; color: #102a43; margin: 42mm 0 8mm; letter-spacing: -0.7pt; }
h2 { font-family: Arial, sans-serif; color: #0b7285; font-size: 17pt; margin: 8mm 0 3mm; border-bottom: 1px solid #9bd4dd; padding-bottom: 1.5mm; page-break-after: avoid; }
h3 { font-family: Arial, sans-serif; color: #174a5b; font-size: 12.5pt; margin: 5mm 0 2mm; page-break-after: avoid; }
p { margin: 0 0 3mm; text-align: justify; }
ul { margin: 1mm 0 3mm 6mm; padding-left: 5mm; }
li { margin-bottom: 1.2mm; text-align: justify; }
.numbered { list-style-type: decimal; }
.title-page { min-height: 245mm; padding: 8mm 8mm 0; position: relative; background: linear-gradient(145deg,#f5fcfd 0%,#ffffff 58%); }
.brand { font-family: Arial,sans-serif; color: #0b7285; font-weight: bold; letter-spacing: 2px; font-size: 10pt; }
.title-rule { width: 42mm; height: 2mm; background: #f59f00; margin: 0 0 14mm; }
.authors { font: bold 14pt Arial,sans-serif; color: #17324d; text-align: left; margin-top: 18mm; }
.affiliation,.date { font: 11pt Arial,sans-serif; color:#526777; text-align:left; }
.cover-flow { position:absolute; left:8mm; right:8mm; bottom:24mm; display:flex; align-items:center; justify-content:space-between; font: bold 10pt Arial,sans-serif; color:#0b7285; }
.cover-flow span { border:1.5px solid #6fc5d1; padding:5mm 4mm; border-radius:4px; background:#fff; }
.cover-flow b { color:#f59f00; font-size:18pt; }
.page-break { page-break-after: always; }
table { width:100%; border-collapse:collapse; margin:3mm 0 5mm; font-family:Arial,sans-serif; font-size:9pt; page-break-inside:avoid; }
th { background:#0b7285; color:white; padding:2.2mm; border:1px solid #087080; text-align:left; }
td { padding:2mm; border:1px solid #bad7dc; vertical-align:top; }
tr:nth-child(even) td { background:#f3fafb; }
pre { font:8.5pt Consolas,monospace; background:#f3f6f8; border-left:4px solid #0b7285; padding:3mm; white-space:pre-wrap; page-break-inside:avoid; }
code { font:9pt Consolas,monospace; background:#eef3f5; padding:0 1mm; }
figure { margin:5mm auto 6mm; text-align:center; page-break-inside:avoid; }
figure img { max-width:92%; max-height:205mm; border:1px solid #d7e1e5; box-shadow:0 1px 4px #bbc7cc; }
figcaption { font:italic 8.5pt Arial,sans-serif; color:#526777; margin-top:2mm; }
.architecture { border:1px solid #b9dce2; background:#f8fdfe; padding:5mm; }
.arch-row { display:flex; justify-content:center; gap:3mm; }
.arch-row.four .arch-box { width:23%; }
.arch-box { font: bold 9.5pt Arial,sans-serif; background:white; color:#174a5b; border:1.5px solid #58b4c2; border-radius:4px; padding:3mm; min-width:45mm; }
.arch-box small { font-weight:normal; color:#607783; }
.arch-box.query { min-width:105mm; border-color:#f59f00; }
.arch-box.fusion { min-width:145mm; background:#e7f7fa; }
.arch-box.output { min-width:125mm; border-color:#2f9e44; }
.arrow { color:#f59f00; font:bold 18pt Arial,sans-serif; line-height:1; margin:1.5mm; }
.source-page { page-break-before:always; }
hr { border:0; border-top:1px solid #aacbd1; margin:6mm 0; }
'@

$html = @"
<!doctype html><html><head><meta charset="utf-8"><title>AIC-TTVN Paper</title><style>$css</style></head>
<body>$($body.ToString())$($appendix.ToString())</body></html>
"@
[System.IO.File]::WriteAllText($htmlPath, $html, [System.Text.UTF8Encoding]::new($false))

$edgeCandidates = @(
    'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
)
$edge = $edgeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $edge) { throw 'Microsoft Edge was not found; cannot render the PDF.' }
$htmlUri = ([uri]$htmlPath).AbsoluteUri
try {
    & $edge --headless --disable-gpu --no-sandbox --allow-file-access-from-files --print-to-pdf=$pdfPath $htmlUri
    if (-not (Test-Path -LiteralPath $pdfPath)) { throw 'PDF rendering did not produce an output file.' }
} finally {
    if (Test-Path -LiteralPath $htmlPath) { Remove-Item -LiteralPath $htmlPath -Force -ErrorAction SilentlyContinue }
}

Write-Output $pdfPath
