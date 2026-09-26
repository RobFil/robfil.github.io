[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('initialize', 'context', 'compare', 'advance', 'skip', 'status', 'version')]
    [string]$Command,

    [string]$InputPath,
    [string]$StatePath,
    [int]$Width = 120,
    [string]$Text
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Utf8Json([object]$Value, [string]$Path) {
    $directory = Split-Path -Parent $Path
    if ($directory) { New-Item -ItemType Directory -Force -Path $directory | Out-Null }
    $json = $Value | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText($Path, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

function Read-State([string]$Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw 'StatePath is required for this command.'
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Reading state not found: $Path"
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-FileSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Result([object]$Value) {
    $Value | ConvertTo-Json -Depth 5 -Compress
}

function Get-ComparableText([string]$Value) {
    return $Value -replace '[\s\u3000\u3001\u3002\uFF01\uFF1F!?\u300C\u300D\u300E\u300F\uFF08\uFF09()]', ''
}

function Get-CurrentSentence([object]$State) {
    $remainder = $State.exact_text.Substring([int]$State.cursor)
    $match = [regex]::Match($remainder, '^.*?[\u3002\uFF01\uFF1F]')
    if ($match.Success) { return $match.Value }
    return $remainder
}

function Get-ConfirmedPrefix([string]$Source, [string]$Transcript) {
    $target = Get-ComparableText $Transcript
    if ([string]::IsNullOrEmpty($target)) { return $null }
    $candidate = ''
    for ($index = 0; $index -lt $Source.Length; $index++) {
        $candidate += Get-ComparableText ([string]$Source[$index])
        if (-not $target.StartsWith($candidate, [System.StringComparison]::Ordinal)) {
            return $null
        }
        if ($candidate -ceq $target) {
            $end = $index + 1
            while ($end -lt $Source.Length -and (Get-ComparableText ([string]$Source[$end])).Length -eq 0) {
                $end++
            }
            return $Source.Substring(0, $end)
        }
    }
    return $null
}

if ($Command -ne 'version' -and [string]::IsNullOrWhiteSpace($StatePath)) {
    throw 'StatePath is required for this command.'
}

switch ($Command) {
    'version' {
        $skillRoot = Split-Path -Parent $PSScriptRoot
        $versionPath = Join-Path $skillRoot 'VERSION'
        $skillPath = Join-Path $skillRoot 'SKILL.md'
        if (-not (Test-Path -LiteralPath $versionPath) -or -not (Test-Path -LiteralPath $skillPath)) {
            throw 'The skill version files are incomplete.'
        }
        Write-Result ([ordered]@{
            skill_version = (Get-Content -LiteralPath $versionPath -Raw -Encoding ascii).Trim()
            skill_sha256 = Get-FileSha256 $skillPath
            script_sha256 = Get-FileSha256 $PSCommandPath
        })
    }
    'initialize' {
        if (-not $InputPath) { throw 'InputPath is required for initialize.' }
        $resolved = (Resolve-Path -LiteralPath $InputPath -ErrorAction Stop).Path
        $raw = Get-Content -LiteralPath $resolved -Raw -Encoding utf8
        $withoutFrontMatter = $raw -replace '\A---\r?\n[\s\S]*?\r?\n---\s*\r?\n', ''
        $body = ($withoutFrontMatter -replace '\r\n?', "`n") -replace "`n## Lernnotiz[\s\S]*\z", ''
        $body = $body.Trim()
        if ([string]::IsNullOrWhiteSpace($body)) { throw 'The post has no readable Japanese body.' }
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $hashBytes = $sha256.ComputeHash($bytes)
        }
        finally {
            $sha256.Dispose()
        }
        $hash = ([System.BitConverter]::ToString($hashBytes) -replace '-', '').ToLowerInvariant()
        $state = [ordered]@{
            version = 1
            source_path = $resolved
            source_sha256 = $hash
            exact_text = $body
            cursor = 0
        }
        Write-Utf8Json $state $StatePath
        Write-Result ([ordered]@{ status = 'initialized'; source_sha256 = $hash; characters = $body.Length; cursor = 0 })
    }
    'context' {
        $state = Read-State $StatePath
        $start = [int]$state.cursor
        $length = [Math]::Min([Math]::Max($Width, 1), $state.exact_text.Length - $start)
        Write-Result ([ordered]@{ cursor = $start; end = $start + $length; text = $state.exact_text.Substring($start, $length) })
    }
    'compare' {
        if ([string]::IsNullOrWhiteSpace($Text)) { throw 'Text is required for compare.' }
        $state = Read-State $StatePath
        $expected = Get-CurrentSentence $state
        $confirmedText = Get-ConfirmedPrefix $expected $Text
        $hasHelpSignal = [regex]::IsMatch($Text, '\u4F55\u3005|\u4F55|\u306A\u306B\u306A\u306B|\u3007\u3007|\u25CB\u25CB|nani\s*nani|nani', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
        if ($null -ne $confirmedText) {
            $decision = 'surface_match'
        }
        elseif ($hasHelpSignal) {
            $decision = 'reading_help_required'
        }
        else {
            $decision = 'review_required'
        }
        $result = [ordered]@{
            decision = $decision
            cursor = [int]$state.cursor
            expected_text = $expected
            confirmed_text = $confirmedText
            transcript = $Text
            source_sha256 = $state.source_sha256
            advance_allowed = ($decision -eq 'surface_match')
        }
        $state | Add-Member -NotePropertyName last_comparison -NotePropertyValue $result -Force
        Write-Utf8Json $state $StatePath
        Write-Result $result
    }
    'advance' {
        if ([string]::IsNullOrEmpty($Text)) { throw 'Text is required for advance.' }
        $state = Read-State $StatePath
        $cursor = [int]$state.cursor
        if ($cursor + $Text.Length -gt $state.exact_text.Length -or
            $state.exact_text.Substring($cursor, $Text.Length) -cne $Text) {
            throw 'Refusing to advance: supplied text is not the exact consecutive source text at the cursor.'
        }
        $state.cursor = $cursor + $Text.Length
        Write-Utf8Json $state $StatePath
        Write-Result ([ordered]@{ status = 'advanced'; cursor = $state.cursor })
    }
    'skip' {
        $state = Read-State $StatePath
        $skippedText = Get-CurrentSentence $state
        if ([string]::IsNullOrEmpty($skippedText)) { throw 'There is no unread source text to skip.' }
        $state.cursor = [int]$state.cursor + $skippedText.Length
        $record = [ordered]@{
            status = 'skipped_unverified'
            cursor = $state.cursor
            skipped_text = $skippedText
        }
        $state | Add-Member -NotePropertyName last_skip -NotePropertyValue $record -Force
        Write-Utf8Json $state $StatePath
        Write-Result $record
    }
    'status' {
        $state = Read-State $StatePath
        Write-Result ([ordered]@{ source_path = $state.source_path; source_sha256 = $state.source_sha256; cursor = $state.cursor; characters = $state.exact_text.Length })
    }
}
