\* model *\
Maximize
OBJ: - batsize - 2 pvsize - 2 zp_0_0 - 2 zp_0_1 - 2 zp_1_0 - 2 zp_1_1
Subject To
cons_bat_low_0_0: - sch_0_0 + sdis_0_0 <= 0
cons_bat_low_0_0_0: - ch_0_0_0 + dis_0_0_0 <= 0
cons_bat_low_0_0_1: - ch_0_0_0 - ch_0_0_1 + dis_0_0_0 + dis_0_0_1 <= 0
cons_bat_low_0_1: - sch_0_0 - sch_0_1 + sdis_0_0 + sdis_0_1 <= 0
cons_bat_low_0_1_0: - ch_0_1_0 + dis_0_1_0 <= 0
cons_bat_low_0_1_1: - ch_0_1_0 - ch_0_1_1 + dis_0_1_0 + dis_0_1_1 <= 0
cons_bat_low_1_0: - sch_1_0 + sdis_1_0 <= 0
cons_bat_low_1_0_0: - ch_1_0_0 + dis_1_0_0 <= 0
cons_bat_low_1_0_1: - ch_1_0_0 - ch_1_0_1 + dis_1_0_0 + dis_1_0_1 <= 0
cons_bat_low_1_1: - sch_1_0 - sch_1_1 + sdis_1_0 + sdis_1_1 <= 0
cons_bat_low_1_1_0: - ch_1_1_0 + dis_1_1_0 <= 0
cons_bat_low_1_1_1: - ch_1_1_0 - ch_1_1_1 + dis_1_1_0 + dis_1_1_1 <= 0
cons_bat_up_0_0_0: ch_0_0_0 - dis_0_0_0 <= 0
cons_bat_up_0_0_1: ch_0_0_0 + ch_0_0_1 - dis_0_0_0 - dis_0_0_1 <= 0
cons_bat_up_0_1_0: ch_0_1_0 - dis_0_1_0 <= 0
cons_bat_up_0_1_1: ch_0_1_0 + ch_0_1_1 - dis_0_1_0 - dis_0_1_1 <= 0
cons_bat_up_1_0_0: ch_1_0_0 - dis_1_0_0 <= 0
cons_bat_up_1_0_1: ch_1_0_0 + ch_1_0_1 - dis_1_0_0 - dis_1_0_1 <= 0
cons_bat_up_1_1_0: ch_1_1_0 - dis_1_1_0 <= 0
cons_bat_up_1_1_1: ch_1_1_0 + ch_1_1_1 - dis_1_1_0 - dis_1_1_1 <= 0
cons_bnd_low_0_0_0: dis_0_0_0 <= 1
cons_bnd_low_0_0_1: dis_0_0_1 <= 1
cons_bnd_low_0_1_0: dis_0_1_0 <= 1
cons_bnd_low_0_1_1: dis_0_1_1 <= 1
cons_bnd_low_1_0_0: dis_1_0_0 <= 1
cons_bnd_low_1_0_1: dis_1_0_1 <= 1
cons_bnd_low_1_1_0: dis_1_1_0 <= 1
cons_bnd_low_1_1_1: dis_1_1_1 <= 1
cons_bnd_up_0_0: - batsize + sch_0_0 <= 0
cons_bnd_up_0_0_0: ch_0_0_0 <= 1
cons_bnd_up_0_0_1: ch_0_0_1 <= 1
cons_bnd_up_0_1: - batsize + sch_0_1 <= 0
cons_bnd_up_0_1_0: ch_0_1_0 <= 1
cons_bnd_up_0_1_1: ch_0_1_1 <= 1
cons_bnd_up_1_0: - batsize + sch_1_0 <= 0
cons_bnd_up_1_0_0: ch_1_0_0 <= 1
cons_bnd_up_1_0_1: ch_1_0_1 <= 1
cons_bnd_up_1_1: - batsize + sch_1_1 <= 0
cons_bnd_up_1_1_0: ch_1_1_0 <= 1
cons_bnd_up_1_1_1: ch_1_1_1 <= 1
cons_shared_bat_up_0_0: - batsize + sch_0_0 - sdis_0_0 <= 0
cons_shared_bat_up_0_1: - batsize + sch_0_0 + sch_0_1 - sdis_0_0 - sdis_0_1
 <= 0
cons_shared_bat_up_1_0: - batsize + sch_1_0 - sdis_1_0 <= 0
cons_shared_bat_up_1_1: - batsize + sch_1_0 + sch_1_1 - sdis_1_0 - sdis_1_1
 <= 0
cons_shared_bnd_low_0_0: - batsize + sdis_0_0 <= 0
cons_shared_bnd_low_0_1: - batsize + sdis_0_1 <= 0
cons_shared_bnd_low_1_0: - batsize + sdis_1_0 <= 0
cons_shared_bnd_low_1_1: - batsize + sdis_1_1 <= 0
cons_z_0_0: - ch_0_0_0 - ch_0_1_0 + dis_0_0_0 + dis_0_1_0 + pvsize - sch_0_0
 + sdis_0_0 - zn_0_0 + zp_0_0 <= 0
cons_z_0_1: - ch_0_0_1 - ch_0_1_1 + dis_0_0_1 + dis_0_1_1 - sch_0_1 + sdis_0_1
 - zn_0_1 + zp_0_1 <= 1
cons_z_1_0: - ch_1_0_0 - ch_1_1_0 + dis_1_0_0 + dis_1_1_0 + pvsize - sch_1_0
 + sdis_1_0 - zn_1_0 + zp_1_0 <= 0
cons_z_1_1: - ch_1_0_1 - ch_1_1_1 + dis_1_0_1 + dis_1_1_1 - sch_1_1 + sdis_1_1
 - zn_1_1 + zp_1_1 <= 1
cons_zo_0_0: ch_0_0_0 + ch_0_1_0 - dis_0_0_0 - dis_0_1_0 - pvsize + sch_0_0
 - sdis_0_0 + zn_0_0 - zp_0_0 <= 0
cons_zo_0_1: ch_0_0_1 + ch_0_1_1 - dis_0_0_1 - dis_0_1_1 + sch_0_1 - sdis_0_1
 + zn_0_1 - zp_0_1 <= -1
cons_zo_1_0: ch_1_0_0 + ch_1_1_0 - dis_1_0_0 - dis_1_1_0 - pvsize + sch_1_0
 - sdis_1_0 + zn_1_0 - zp_1_0 <= 0
cons_zo_1_1: ch_1_0_1 + ch_1_1_1 - dis_1_0_1 - dis_1_1_1 + sch_1_1 - sdis_1_1
 + zn_1_1 - zp_1_1 <= -1
End
