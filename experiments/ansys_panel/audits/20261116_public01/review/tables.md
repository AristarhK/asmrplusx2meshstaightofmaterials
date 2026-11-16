# Saved panel study: numerical review

VM means in MPa. C=(500,300), Y=(500,48), X=(48,300); radius8 mm.

## P0: original linear means

| h mm | TOP C | TOP Y | TOP X | BOTTOM C | BOTTOM Y | BOTTOM X |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 12.981456365 | 3.973260382 | 2.788839727 | 12.981456365 | 3.973260382 | 2.788839727 |
| 25 | 13.063383419 | 4.034552540 | 2.883909264 | 13.063383419 | 4.034552540 | 2.883909264 |
| 12.5 | 13.102335963 | 4.098805530 | 2.938519836 | 13.102335963 | 4.098805530 | 2.938519836 |
| 6.25 | 13.126942795 | 4.129090379 | 2.971593317 | 13.126942795 | 4.129090379 | 2.971593317 |
| 3.125 | 13.138040674 | 4.141356959 | 2.986539183 | 13.138040674 | 4.141356959 | 2.986539183 |

## P1: original linear means

| h mm | TOP C | TOP Y | TOP X | BOTTOM C | BOTTOM Y | BOTTOM X |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 5.591167023 | 6.484436794 | 4.316539866 | 5.591167023 | 6.484436794 | 4.316539866 |
| 25 | 5.659607485 | 6.419379147 | 4.181086599 | 5.659607485 | 6.419379147 | 4.181086599 |
| 12.5 | 5.680490607 | 6.350800469 | 4.100075802 | 5.680490607 | 6.350800469 | 4.100075802 |
| 6.25 | 5.687544718 | 6.331779728 | 4.077914041 | 5.687544718 | 6.331779728 | 4.077914041 |
| 3.125 | 5.689230833 | 6.327003108 | 4.072345195 | 5.689230833 | 6.327003108 | 4.072345195 |

## Last two mesh transitions, percentages

| Support | Surface | Disk | 12.5→6.25 % | 6.25→3.125 % | Both ≤1% |
|---|---|---|---:|---:|---|
| P0 | top | center | 0.187453 | 0.084471 | yes |
| P0 | top | near_y_edge | 0.733451 | 0.296197 | yes |
| P0 | top | near_x_edge | 1.112988 | 0.500441 | NO |
| P0 | bottom | center | 0.187453 | 0.084471 | yes |
| P0 | bottom | near_y_edge | 0.733451 | 0.296197 | yes |
| P0 | bottom | near_x_edge | 1.112988 | 0.500441 | NO |
| P1 | top | center | 0.124027 | 0.029637 | yes |
| P1 | top | near_y_edge | 0.300401 | 0.075496 | yes |
| P1 | top | near_x_edge | 0.543458 | 0.136748 | yes |
| P1 | bottom | center | 0.124027 | 0.029637 | yes |
| P1 | bottom | near_y_edge | 0.300401 | 0.075496 | yes |
| P1 | bottom | near_x_edge | 0.543458 | 0.136748 | yes |

## P1 linear vs nonlinear on h=3.125 mm

| Surface | Disk | Linear MPa | Nonlinear MPa | Δ % / NL | Δ % / linear | ≤2% |
|---|---|---:|---:|---:|---:|---|
| top | center | 5.689230833 | 5.721922126 | 0.571334 | 0.574617 | yes |
| top | near_y_edge | 6.327003108 | 6.291823372 | 0.559134 | 0.556025 | yes |
| top | near_x_edge | 4.072345195 | 4.057680173 | 0.361414 | 0.360112 | yes |
| bottom | center | 5.689230833 | 5.653463998 | 0.632653 | 0.628676 | yes |
| bottom | near_y_edge | 6.327003108 | 6.359377633 | 0.509083 | 0.511688 | yes |
| bottom | near_x_edge | 4.072345195 | 4.085813392 | 0.329633 | 0.330723 | yes |

## References by predefined metric

| Support | Metric | Earlier Δ % | Last Δ % | Limit % | Stable twice |
|---|---|---:|---:|---:|---|
| P0 | central_deflection | 0.188619 | 0.094400 | 0.2 | yes |
| P0 | max_abs_uz | 0.188619 | 0.094400 | 0.2 | yes |
| P0 | strain_energy | 0.264449 | 0.122339 | 0.5 | yes |
| P0 | top/center/mean_von_mises | 0.187453 | 0.084471 | 1 | yes |
| P0 | top/near_y_edge/mean_von_mises | 0.733451 | 0.296197 | 1 | yes |
| P0 | top/near_x_edge/mean_von_mises | 1.112988 | 0.500441 | 1 | NO |
| P0 | bottom/center/mean_von_mises | 0.187453 | 0.084471 | 1 | yes |
| P0 | bottom/near_y_edge/mean_von_mises | 0.733451 | 0.296197 | 1 | yes |
| P0 | bottom/near_x_edge/mean_von_mises | 1.112988 | 0.500441 | 1 | NO |
| P1 | central_deflection | 0.072764 | 0.018203 | 0.2 | yes |
| P1 | max_abs_uz | 0.072764 | 0.018203 | 0.2 | yes |
| P1 | strain_energy | 0.126711 | 0.031716 | 0.5 | yes |
| P1 | top/center/mean_von_mises | 0.124027 | 0.029637 | 1 | yes |
| P1 | top/near_y_edge/mean_von_mises | 0.300401 | 0.075496 | 1 | yes |
| P1 | top/near_x_edge/mean_von_mises | 0.543458 | 0.136748 | 1 | yes |
| P1 | bottom/center/mean_von_mises | 0.124027 | 0.029637 | 1 | yes |
| P1 | bottom/near_y_edge/mean_von_mises | 0.300401 | 0.075496 | 1 | yes |
| P1 | bottom/near_x_edge/mean_von_mises | 0.543458 | 0.136748 | 1 | yes |
