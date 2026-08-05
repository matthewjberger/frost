# What the edit-compile loop costs, re-runnable.
#
# The thesis is that a question is cheaper to answer by probe than by reasoning,
# and that is a claim about wall time. These are the numbers behind it. Run from
# the repository root:
#
#   just bench-loop
#
# Every figure is a median of five runs after one that is thrown away, since the
# first touches a cold file cache and the object cache the runtime keeps.

param([int]$Runs = 5)

function Median-Ms {
    param([scriptblock]$Body, [int]$Times)
    $null = & $Body
    $taken = @()
    for ($i = 0; $i -lt $Times; $i++) {
        $taken += (Measure-Command { & $Body }).TotalMilliseconds
    }
    ($taken | Sort-Object)[[int]($taken.Count / 2)]
}

$frost = "./target/release/frost.exe"
$hosted = "./selfhosted/frost.exe"
$tmp = $env:TEMP
$one = "examples/native/slices.frost"
$big = "selfhosted/regions.frost"

"hardware: $((Get-CimInstance Win32_Processor).Name.Trim()), $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)) GB"
"commit:   $(git rev-parse --short HEAD)"
""
"{0,-44} {1,9}" -f "measurement", "median ms"
"{0,-44} {1,9}" -f ("-" * 44), ("-" * 9)

$cold = Median-Ms { & $frost --native -o "$tmp/loop.o" $one 2>&1 | Out-Null } $Runs
"{0,-44} {1,9:N0}" -f "one file, checked to an object", $cold

$test = Median-Ms { & $frost --test $one 2>&1 | Out-Null } $Runs
"{0,-44} {1,9:N0}" -f "one file, compiled and its tests run", $test

$module = Median-Ms { & $frost --native -o "$tmp/loop.o" $big 2>&1 | Out-Null } $Runs
"{0,-44} {1,9:N0}" -f "largest module, checked to an object", $module

$fmt = Median-Ms { & $frost fmt --check std lib selfhosted examples 2>&1 | Out-Null } $Runs
"{0,-44} {1,9:N0}" -f "frost fmt --check over the corpus", $fmt

$lint = Median-Ms { & $frost lint std 2>&1 | Out-Null } $Runs
"{0,-44} {1,9:N0}" -f "frost lint over the standard library", $lint

$api = Median-Ms { & $frost api vec_ std 2>&1 | Out-Null } $Runs
"{0,-44} {1,9:N0}" -f "frost api over the standard library", $api

# The fixpoint is the whole compiler twice and is measured once: five runs of it
# is minutes of nothing new.
$fix = (Measure-Command {
    & $frost --link --emit-c -o "$tmp/loop_gen1.exe" selfhosted/frost.frost 2>&1 | Out-Null
    $env:FROST_INPUT = "selfhosted/frost.frost"
    & "$tmp/loop_gen1.exe" -o "$tmp/loop_gen2.c" 2>&1 | Out-Null
    $env:FROST_INPUT = $null
}).TotalMilliseconds
"{0,-44} {1,9:N0}" -f "self-hosting fixpoint, both generations", $fix
