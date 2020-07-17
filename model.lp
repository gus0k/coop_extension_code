\* model *\
Maximize
OBJ: - batsizeXX - 2 pvsizeXX + 0.05 znXX0_0 + 0.05 znXX0_1 + 0.05 znXX1_0
 + 0.05 znXX1_1 - 2 zpXX0_0 - 2 zpXX0_1 - 2 zpXX1_0 - 2 zpXX1_1
Subject To
cons_bat_low_0_0_0: - chXX0_0_0 + disXX0_0_0 <= 0
cons_bat_low_0_0_1: - chXX0_0_0 - chXX0_0_1 + disXX0_0_0 + disXX0_0_1 <= 0
cons_bat_low_0_1_0: - chXX0_1_0 + disXX0_1_0 <= 0
cons_bat_low_0_1_1: - chXX0_1_0 - chXX0_1_1 + disXX0_1_0 + disXX0_1_1 <= 0
cons_bat_low_1_0_0: - chXX1_0_0 + disXX1_0_0 <= 0
cons_bat_low_1_0_1: - chXX1_0_0 - chXX1_0_1 + disXX1_0_0 + disXX1_0_1 <= 0
cons_bat_low_1_1_0: - chXX1_1_0 + disXX1_1_0 <= 0
cons_bat_low_1_1_1: - chXX1_1_0 - chXX1_1_1 + disXX1_1_0 + disXX1_1_1 <= 0
cons_bat_up_0_0_0: chXX0_0_0 - disXX0_0_0 <= 0
cons_bat_up_0_0_1: chXX0_0_0 + chXX0_0_1 - disXX0_0_0 - disXX0_0_1 <= 0
cons_bat_up_0_1_0: chXX0_1_0 - disXX0_1_0 <= 0
cons_bat_up_0_1_1: chXX0_1_0 + chXX0_1_1 - disXX0_1_0 - disXX0_1_1 <= 0
cons_bat_up_1_0_0: chXX1_0_0 - disXX1_0_0 <= 0
cons_bat_up_1_0_1: chXX1_0_0 + chXX1_0_1 - disXX1_0_0 - disXX1_0_1 <= 0
cons_bat_up_1_1_0: chXX1_1_0 - disXX1_1_0 <= 0
cons_bat_up_1_1_1: chXX1_1_0 + chXX1_1_1 - disXX1_1_0 - disXX1_1_1 <= 0
cons_bnd_low_0_0_0: disXX0_0_0 <= 1
cons_bnd_low_0_0_1: disXX0_0_1 <= 1
cons_bnd_low_0_1_0: disXX0_1_0 <= 1
cons_bnd_low_0_1_1: disXX0_1_1 <= 1
cons_bnd_low_1_0_0: disXX1_0_0 <= 1
cons_bnd_low_1_0_1: disXX1_0_1 <= 1
cons_bnd_low_1_1_0: disXX1_1_0 <= 1
cons_bnd_low_1_1_1: disXX1_1_1 <= 1
cons_bnd_up_0_0_0: chXX0_0_0 <= 1
cons_bnd_up_0_0_1: chXX0_0_1 <= 1
cons_bnd_up_0_1_0: chXX0_1_0 <= 1
cons_bnd_up_0_1_1: chXX0_1_1 <= 1
cons_bnd_up_1_0_0: chXX1_0_0 <= 1
cons_bnd_up_1_0_1: chXX1_0_1 <= 1
cons_bnd_up_1_1_0: chXX1_1_0 <= 1
cons_bnd_up_1_1_1: chXX1_1_1 <= 1
cons_shared_bat_low_0_0: - schXX0_0 + sdisXX0_0 <= 0
cons_shared_bat_low_0_1: - schXX0_0 - schXX0_1 + sdisXX0_0 + sdisXX0_1 <= 0
cons_shared_bat_low_1_0: - schXX1_0 + sdisXX1_0 <= 0
cons_shared_bat_low_1_1: - schXX1_0 - schXX1_1 + sdisXX1_0 + sdisXX1_1 <= 0
cons_shared_bat_up_0_0: - batsizeXX + schXX0_0 - sdisXX0_0 <= 0
cons_shared_bat_up_0_1: - batsizeXX + schXX0_0 + schXX0_1 - sdisXX0_0
 - sdisXX0_1 <= 0
cons_shared_bat_up_1_0: - batsizeXX + schXX1_0 - sdisXX1_0 <= 0
cons_shared_bat_up_1_1: - batsizeXX + schXX1_0 + schXX1_1 - sdisXX1_0
 - sdisXX1_1 <= 0
cons_shared_bnd_low_0_0: - batsizeXX + sdisXX0_0 <= 0
cons_shared_bnd_low_0_1: - batsizeXX + sdisXX0_1 <= 0
cons_shared_bnd_low_1_0: - batsizeXX + sdisXX1_0 <= 0
cons_shared_bnd_low_1_1: - batsizeXX + sdisXX1_1 <= 0
cons_shared_bnd_up_0_0: - batsizeXX + schXX0_0 <= 0
cons_shared_bnd_up_0_1: - batsizeXX + schXX0_1 <= 0
cons_shared_bnd_up_1_0: - batsizeXX + schXX1_0 <= 0
cons_shared_bnd_up_1_1: - batsizeXX + schXX1_1 <= 0
cons_z_0_0: - chXX0_0_0 - chXX0_1_0 + disXX0_0_0 + disXX0_1_0 + pvsizeXX
 - schXX0_0 + sdisXX0_0 - znXX0_0 + zpXX0_0 <= 0
cons_z_0_1: - chXX0_0_1 - chXX0_1_1 + disXX0_0_1 + disXX0_1_1 - schXX0_1
 + sdisXX0_1 - znXX0_1 + zpXX0_1 <= 1.2
cons_z_1_0: - chXX1_0_0 - chXX1_1_0 + disXX1_0_0 + disXX1_1_0 + pvsizeXX
 - schXX1_0 + sdisXX1_0 - znXX1_0 + zpXX1_0 <= 0
cons_z_1_1: - chXX1_0_1 - chXX1_1_1 + disXX1_0_1 + disXX1_1_1 - schXX1_1
 + sdisXX1_1 - znXX1_1 + zpXX1_1 <= 1.5
cons_zo_0_0: chXX0_0_0 + chXX0_1_0 - disXX0_0_0 - disXX0_1_0 - pvsizeXX
 + schXX0_0 - sdisXX0_0 + znXX0_0 - zpXX0_0 <= 0
cons_zo_0_1: chXX0_0_1 + chXX0_1_1 - disXX0_0_1 - disXX0_1_1 + schXX0_1
 - sdisXX0_1 + znXX0_1 - zpXX0_1 <= -1.2
cons_zo_1_0: chXX1_0_0 + chXX1_1_0 - disXX1_0_0 - disXX1_1_0 - pvsizeXX
 + schXX1_0 - sdisXX1_0 + znXX1_0 - zpXX1_0 <= 0
cons_zo_1_1: chXX1_0_1 + chXX1_1_1 - disXX1_0_1 - disXX1_1_1 + schXX1_1
 - sdisXX1_1 + znXX1_1 - zpXX1_1 <= -1.5
End
