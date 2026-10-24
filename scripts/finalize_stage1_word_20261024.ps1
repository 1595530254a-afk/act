param(
    [Parameter(Mandatory = $true)][string]$BaseDocx,
    [Parameter(Mandatory = $true)][string]$OutputDocx,
    [Parameter(Mandatory = $true)][string]$MediaManifest,
    [Parameter(Mandatory = $true)][string]$PdfOutput
)

# Work on a new SaveAs copy. Never write to the base or an existing output.
$ErrorActionPreference = 'Stop'
$basePath = (Get-Item -LiteralPath $BaseDocx).FullName
$manifestPath = (Get-Item -LiteralPath $MediaManifest).FullName
$outputPath = [IO.Path]::GetFullPath($OutputDocx)
$pdfPath = [IO.Path]::GetFullPath($PdfOutput)
if ([IO.Path]::GetExtension($basePath) -ne '.docx' -or
    [IO.Path]::GetExtension($outputPath) -ne '.docx' -or
    [IO.Path]::GetExtension($pdfPath) -ne '.pdf') {
    throw 'Expected a base DOCX, a new output DOCX and a new PDF path.'
}
if ($basePath -eq $outputPath -or (Test-Path -LiteralPath $outputPath) -or
    (Test-Path -LiteralPath $pdfPath)) {
    throw 'Preserve the base and earlier outputs; choose new DOCX/PDF paths.'
}
$baseHash = (Get-FileHash -LiteralPath $basePath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$raw = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($raw.PSObject.Properties.Name -contains 'entries') { $entries = @($raw.entries) }
else { $entries = @($raw) }
if ($entries.Count -eq 0) { throw 'Media manifest is empty.' }
$seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$media = @()
foreach ($entry in $entries) {
    $id = [string]$entry.id
    $expectedHash = [string]$entry.sha256
    if ($id -notmatch '^[A-Za-z0-9_-]+$' -or -not $seen.Add($id) -or
        $expectedHash -notmatch '^[a-fA-F0-9]{64}$' -or -not $entry.path) {
        throw 'Each media entry needs a unique marker ID, path and SHA256.'
    }
    $candidate = [string]$entry.path
    if (-not [IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path (Split-Path -Parent $manifestPath) $candidate
    }
    $path = (Get-Item -LiteralPath $candidate).FullName
    if ([IO.Path]::GetExtension($path) -ne '.mp4') { throw "Expected MP4 for $id" }
    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash.ToLowerInvariant()) { throw "MP4 SHA mismatch: $id" }
    $media += [PSCustomObject]@{ id = $id; path = $path; sha256 = $actualHash }
}
foreach ($parent in @((Split-Path -Parent $outputPath), (Split-Path -Parent $pdfPath))) {
    [void][IO.Directory]::CreateDirectory($parent)
}

$app = $null
$doc = $null
$initialCount = -1
$embedded = 0
try {
    $app = New-Object -ComObject kwps.Application
    $initialCount = [int]$app.Documents.Count
    if ($initialCount -eq 0) { $app.Visible = $false }
    # COM can reuse an already-open document. Refuse that case so closing our
    # document cannot close a user's pre-existing view of the base.
    for ($index = 1; $index -le $initialCount; $index++) {
        $existing = $app.Documents.Item($index)
        try {
            if ([string]$existing.FullName -eq $basePath) {
                throw 'The base is already open in Office; this tool will not take ownership of that document.'
            }
        } finally {
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($existing)
        }
    }
    $doc = $app.Documents.Open($basePath, $false, $true, $false)
    $text = [string]$doc.Content.Text
    $markers = [regex]::Matches($text, '\[\[VIDEO:([A-Za-z0-9_-]+)\]\]')
    if ($markers.Count -ne $media.Count) { throw 'Marker count differs from media manifest.' }
    foreach ($entry in $media) {
        $marker = '[[VIDEO:' + $entry.id + ']]'
        if ([regex]::Matches($text, [regex]::Escape($marker)).Count -ne 1) {
            throw "Expected exactly one marker: $marker"
        }
    }
    # Save the text/figures to the new path before creating any OLE objects.
    $doc.SaveAs2($outputPath, 16)
    foreach ($entry in $media) {
        $marker = '[[VIDEO:' + $entry.id + ']]'
        $range = $doc.Content.Duplicate
        $find = $range.Find
        $object = $null
        try {
            $find.ClearFormatting()
            $find.Text = $marker
            $find.Forward = $true
            $find.Wrap = 0
            $find.Format = $false
            $find.MatchCase = $true
            $find.MatchWholeWord = $false
            $find.MatchWildcards = $false
            if (-not $find.Execute()) { throw "Marker disappeared: $marker" }
            $paragraphText = ([string]$range.Paragraphs.Item(1).Range.Text).Trim([char[]]"`r`n`t `a")
            if ($paragraphText -ne $marker) { throw "Marker must occupy its own paragraph: $marker" }
            $range.Text = ''
            $range.Collapse(1)
            $missing = [Type]::Missing
            $object = $doc.InlineShapes.AddOLEObject('Package', $entry.path, $false, $true,
                $missing, $missing, ($entry.id + '.mp4'), $range)
            $embedded++
        } finally {
            if ($null -ne $object) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($object) }
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($find)
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($range)
        }
    }
    if (([string]$doc.Content.Text).Contains('[[VIDEO:')) { throw 'Unresolved VIDEO marker remains.' }
    $doc.Save()
    $doc.ExportAsFixedFormat($pdfPath, 17)
    foreach ($entry in $media) {
        if ((Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
            throw ('Source MP4 changed during embedding: ' + $entry.id)
        }
    }
    if ((Get-FileHash -LiteralPath $basePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $baseHash -or
        (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifestHash) {
        throw 'Base or media manifest changed during finalization.'
    }
    [PSCustomObject]@{ status = 'office_saved_embedded_copy_and_pdf_requires_payload_verification'
        docx = $outputPath; pdf = $pdfPath; embedded_objects_added = $embedded
        manifest_entries = $media.Count; initial_open_documents = $initialCount
        base_sha256 = $baseHash; base_unchanged = $true
        manifest_sha256 = $manifestHash
        note = 'PDF shows OLE icons; complete MP4 payloads belong to the DOCX and adjacent media files.'
    } | ConvertTo-Json -Depth 4
} finally {
    # Do not save unsaved partial edits when anything failed. The earlier
    # SaveAs copy, if present, stays available for diagnosis; the base is read-only.
    if ($null -ne $doc) {
        try { $doc.Close(0) }
        finally { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc) }
    }
    if ($null -ne $app) {
        try {
            if ($initialCount -eq 0 -and $app.Documents.Count -eq 0) { $app.Quit() }
        } finally {
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)
        }
    }
}
