"""Read the native 2026 R1 CDB load encodings used by the mesh-only gate.

This deliberately narrow parser fails closed on unfamiliar commands/formats. It
does not execute APDL. Geometry blocks are ignored; the independent mesh export
is responsible for geometry. Duplicate load records are retained for the caller
to reject. Returned records contain strings, compatible with APDL command fields.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re


NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?\Z")
DOFS = {"UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"}
GEOMETRY_BLOCKS = {"ETBLOCK", "NBLOCK", "EBLOCK", "CMBLOCK", "SECBLOCK", "RLBLOCK"}
OTHER_BLOCKS = {"FBLOCK", "SFBLOCK", "BFBLOCK", "BFEBLOCK", "ICBLOCK", "CPBLOCK", "CEBLOCK"}
OTHER_LOADS = {
    "F", "SF", "BF", "BFE", "IC", "CP", "CE", "CPDATA", "CPCYC", "CPINTF",
    "CEINTF", "CERIG", "DDELE", "FDELE", "SFDELE", "SFEDELE", "BFDELE",
    "BFEDELE", "ICDELE", "CPDELE", "CEDELE", "DK", "DL", "DA", "FK", "SFL",
    "SFA", "BFK", "BFL", "BFA", "BFV", "SFCONTROL", "SFGRAD", "SFFUN",
    "BFCUM", "BFECUM", "SFCUM", "FCUM", "DCUM", "INISTATE", "LDREAD",
}
ZERO_INERTIAL = {"ACEL", "OMEGA", "DOMEGA", "CGLOC", "CGOMEGA", "DCGOMG"}
# Only metadata observed in the native CDWRITE DB export, plus its unblocked
# geometry equivalents. New formats need an explicit review, never a silent skip.
METADATA = {
    "/COM", "/PREP7", "/NOPR", "/TITLE", "/GO", "/GOPR", "FINISH", "*IF",
    "*ELSE", "*ENDIF", "*SET", "NUMOFF", "DOF", "ANTYPE", "EQSLV", "ET",
    "KEYOPT", "ETCONTROL", "N", "E", "EN", "TYPE", "MAT", "SECNUM", "REAL",
    "ESYS", "MP", "MPTEMP", "MPDATA", "SECTYPE", "SECOFFSET", "SECCONTROL",
    "EXTOPT", "TREF", "IRLF", "KUSE", "TIME", "ALPHAD", "BETAD", "DMPRAT",
    "DMPSTR", "CRPLIM", "NCNV", "NEQIT", "ERESX", "CM", "CMSEL", "ALLSEL",
}


def _number(value: str, context: str) -> str:
    value = value.strip()
    if not NUMBER.fullmatch(value):
        raise ValueError(f"{context}: unsupported numeric value {value!r}")
    result = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(result):
        raise ValueError(f"{context}: nonfinite numeric value")
    return format(result, ".17g")


def _positive_id(value: str, context: str) -> str:
    if not re.fullmatch(r"\s*\d+\s*", value) or int(value) <= 0:
        raise ValueError(f"{context}: expected a positive integer ID, got {value!r}")
    return str(int(value))


def _format(line: str) -> str:
    return re.sub(r"\s+", "", line).lower()


def _fields(line: str) -> list[str]:
    return [part.strip() for part in line.split("!", 1)[0].split(",")]


def _sentinel(fields: list[str]) -> bool:
    return (len(fields) >= 4 and [x.upper() for x in fields[:3]] == ["SFE", "END", "LOC"]
            and fields[3] == "-1" and not any(fields[4:]))


def _fixed(line: str, widths: tuple[int, ...], context: str) -> list[str]:
    size = sum(widths)
    if line[size:].strip():
        raise ValueError(f"{context}: unexpected trailing fixed-width data")
    result = []
    offset = 0
    for width in widths:
        result.append(line[offset:offset + width].strip())
        offset += width
    return result


def _d_record(fields: list[str], context: str) -> list[str]:
    if len(fields) < 4 or len(fields) > 5 or fields[2].upper() not in DOFS:
        raise ValueError(f"{context}: unsupported D record")
    return ["D", _positive_id(fields[1], context), fields[2].upper(),
            _number(fields[3] or "0", context),
            _number(fields[4] or "0", context) if len(fields) == 5 else "0"]


def _sfe_record(fields: list[str], context: str) -> list[str] | None:
    if not 6 <= len(fields) <= 10 or fields[3].upper() != "PRES":
        raise ValueError(f"{context}: unsupported SFE record")
    if len(fields) == 10 and fields[9] not in ("", "0"):
        raise ValueError(f"{context}: unsupported SFE mesh flag")
    elem = _positive_id(fields[1], context)
    face = _positive_id(fields[2] or "1", context)
    kval = fields[4] or "0"
    if kval not in ("0", "1", "2"):
        raise ValueError(f"{context}: unsupported pressure KVAL {kval!r}")
    raw = (fields[5:9] + [""] * 4)[:4]
    first = _number(raw[0] or "0", context)
    values = [first] * 4 if not any(raw[1:]) else [first] + [_number(x or "0", context) for x in raw[1:]]
    if kval == "2":
        return ["SFE_IMAG", elem, face, "PRES", "2", *values] if any(float(x) for x in values) else None
    return ["SFE", elem, face, "PRES", "1", *values]


def parse_cdb_loads(path: str | Path) -> list[list[str]]:
    """Normalize D/DBLOCK and SFE/SFEBLOCK; flag all other known loads.

    D records are ``D,node,DOF,real,imag``. SFE records are
    ``SFE,element,face,PRES,1,p1,p2,p3,p4``. Nonzero imaginary pressure is an
    ``SFE_IMAG`` record, which must fail this static mesh gate. Zero imaginary
    pressure records are checked then omitted. Unknown records raise ValueError.
    """
    lines = Path(path).read_text(encoding="utf-8-sig", errors="strict").splitlines()
    records: list[list[str]] = []
    index = 0
    geometry_payload = False
    while index < len(lines):
        line = lines[index]
        index += 1
        stripped = line.strip()
        context = f"{Path(path).name}:{index}"
        if not stripped or stripped.startswith("!"):
            continue
        fields = _fields(line)
        command = fields[0].upper()
        if command == "DBLOCK":
            geometry_payload = False
            if len(fields) != 2 or not fields[1].isdigit() or index >= len(lines):
                raise ValueError(f"{context}: unsupported DBLOCK header")
            count = int(fields[1])
            if _format(lines[index]) != "(i9,a6,2(pg16.9))":
                raise ValueError(f"{context}: unsupported DBLOCK format")
            index += 1
            for _ in range(count):
                if index >= len(lines):
                    raise ValueError(f"{context}: truncated DBLOCK")
                data = _fixed(lines[index], (9, 6, 16, 16), context)
                records.append(_d_record(["D", *data], context))
                index += 1
            if index >= len(lines) or lines[index].strip() != "-1":
                raise ValueError(f"{context}: DBLOCK count/terminator mismatch")
            index += 1
        elif command == "SFEBLOCK":
            geometry_payload = False
            if (len(fields) != 6 or fields[1:3] != ["4", "PRES"]
                    or not fields[3].isdigit() or not fields[4].isdigit() or fields[5] != "0"):
                raise ValueError(f"{context}: unsupported SFEBLOCK header")
            expected = int(fields[3])
            if index >= len(lines) or _format(lines[index]) != "(i9,i4,i4,6(pg20.9))":
                raise ValueError(f"{context}: unsupported SFEBLOCK format")
            index += 1
            real, imag = Counter(), Counter()
            while index < len(lines) and not _sentinel(_fields(lines[index])):
                data = _fixed(lines[index], (9, 4, 4, 20, 20, 20, 20), context)
                elem, face = _positive_id(data[0], context), _positive_id(data[1], context)
                if data[2] not in ("1", "2") or any(not value for value in data[3:]):
                    raise ValueError(f"{context}: unsupported SFEBLOCK data")
                values = [_number(value, context) for value in data[3:]]
                target = real if data[2] == "1" else imag
                target[(elem, face)] += 1
                record = _sfe_record(["SFE", elem, face, "PRES", data[2], *values], context)
                if record is not None:
                    records.append(record)
                index += 1
            if index >= len(lines):
                raise ValueError(f"{context}: missing SFEBLOCK terminator")
            index += 1
            if sum(real.values()) != expected or real != imag:
                raise ValueError(f"{context}: SFEBLOCK real/imag/count mismatch")
        elif command in GEOMETRY_BLOCKS:
            geometry_payload = True
        elif command in OTHER_BLOCKS:
            records.append([command, *fields[1:]])
            geometry_payload = True
        elif re.match(r"^[A-Za-z/*]", stripped):
            geometry_payload = False
            if command == "D":
                records.append(_d_record(fields, context))
            elif command == "SFE":
                if not _sentinel(fields):
                    record = _sfe_record(fields, context)
                    if record is not None:
                        records.append(record)
            elif command in OTHER_LOADS:
                records.append([command, *fields[1:]])
            elif command == "BFUNIF":
                if [x.upper() for x in fields[1:]] != ["TEMP", "_TINY"]:
                    records.append([command, *fields[1:]])
            elif command in ZERO_INERTIAL:
                values = [_number(x, context) for x in fields[1:] if x]
                if any(float(x) for x in values):
                    records.append([command, *values])
            elif command not in METADATA:
                raise ValueError(f"{context}: unknown CDB command/block {command!r}")
        elif "=" in stripped and re.match(r"^_[A-Za-z0-9_]+\s*=\s*$", stripped.split("!", 1)[0]):
            geometry_payload = False  # Native _CDRDOFF= reset statement.
        elif not geometry_payload:
            raise ValueError(f"{context}: unexpected data outside a recognized block")
    return records
