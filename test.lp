\* model *\
Maximize
OBJ: zn_0 + zn_1 - 3 zp_0 - 3 zp_1
Subject To
cons_bat_low_0_0: - ch_0_0 + dis_0_0 <= 0
cons_bat_low_0_1: - ch_0_0 - ch_0_1 + dis_0_0 + dis_0_1 <= 0
cons_bat_low_1_0: - ch_1_0 + dis_1_0 <= 0
cons_bat_low_1_1: - ch_1_0 - ch_1_1 + dis_1_0 + dis_1_1 <= 0
cons_bat_up_0_0: ch_0_0 - dis_0_0 <= 0
cons_bat_up_0_1: ch_0_0 + ch_0_1 - dis_0_0 - dis_0_1 <= 0
cons_bat_up_1_0: ch_1_0 - dis_1_0 <= 1
cons_bat_up_1_1: ch_1_0 + ch_1_1 - dis_1_0 - dis_1_1 <= 1
cons_bnd_low_0_0: dis_0_0 <= 1
cons_bnd_low_0_1: dis_0_1 <= 1
cons_bnd_low_1_0: dis_1_0 <= 1
cons_bnd_low_1_1: dis_1_1 <= 1
cons_bnd_up_0_0: ch_0_0 <= 1
cons_bnd_up_0_1: ch_0_1 <= 1
cons_bnd_up_1_0: ch_1_0 <= 1
cons_bnd_up_1_1: ch_1_1 <= 1
cons_z_0: - 1.11111111111 ch_0_0 - 1.11111111111 ch_1_0 + 0.9 dis_0_0
 + 0.9 dis_1_0 - zn_0 + zp_0 <= 0
cons_z_1: - 1.11111111111 ch_0_1 - 1.11111111111 ch_1_1 + 0.9 dis_0_1
 + 0.9 dis_1_1 - zn_1 + zp_1 <= 3
cons_zo_0: 1.11111111111 ch_0_0 + 1.11111111111 ch_1_0 - 0.9 dis_0_0
 - 0.9 dis_1_0 + zn_0 - zp_0 <= 0
cons_zo_1: 1.11111111111 ch_0_1 + 1.11111111111 ch_1_1 - 0.9 dis_0_1
 - 0.9 dis_1_1 + zn_1 - zp_1 <= -3
End
