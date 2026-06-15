"""
Contains book split lists for shakespeare 3rd/4th folio experiments.
You can pass this script file contents through stdin and the desired folio number filter,
and it will output to stdout the list of filenames with the book name in the name.
"""


folio3and4 = [
    "amaxwell_R153_uk_4_englishschoolmaster1673",
    "amaxwell_R8130_uk_2_choiceandpracticalexpositions1675",
    "anon_R25621_usppic_2_shakespeare4folio1685",
    "anon_R30560_usppic_2_shakespeare3folio1664",
    "daniel_R12045_ncaotu_8_opusculavaria1658",
    "daniel_R226090_uk_2_holywarre1647",
    "everingham_R3105_de12_4_ogygia1685",
    "everingham_R771_de12_2_someyearstravels1677",
    "hayes_R35661_spmauc_4_deratione1664",
    "hayes_R8679_de12_4_opticapromota1663",
    "hodgkinson_R34604_4_uscu_natureofprocurations1661",
    "hodgkinson_R7102_4_usnnc_restauranda1662",
    "jhayes_R216914_uk_8_aconsolatorydiscourse1665",
    "jhayes_R217129_uk_8_abriefexhortation1665",
    "leach_R217001_uk_4_thehistoryoftarquin1669",
    "macock_R10405_ustxhr_4_manofmode1684",
    "macock_R25438_usnnut_4_gentiles1677",
    "macock_R25855_usmiu_4_ephemerides1672",
    "macock_xxxx_de12_4_parsonswedding1663",
    "maxwellroberts_R10994_usnnut_4_gentiles1677",
    "maxwellroberts_R24847_usmb_4_scornfullady1677",
    "ratcliffe_R11702_usnjpt_4_treatiseminister1670",
    "ratcliffe_R203411_uscua_4_meansandmethod1660",
    "ratcliffe_R618_gbuklw_4_sermonpreached1666",
    "rawlins_R23326_uscstmogri_2_musaeumregalis1681",
    "tratcliffe_R20427_uk_4_thepastoraloffice1663",
    "tratcliffe_R22215_uk_4_thepeaceofjerusalem1659",
    "tratcliffe_R5572_uk_4_romeinherfruits1663",
    "tratcliffenthompson_R212964_uk_4_greatbritainsglory1672",
    "warren_R15965_uk_4_constitutionsandcanons1662",
    "warren_R230847_uk_4_summeandsubstance1661",
    # missing_books.txt: accidentally skipped these in first alignment run. TODO: reverse applying saved models...
    "leach_xxxxx_yyyyy_00height_acoppy",
    "macock_xxxxx_yyyyy_00height_diodorus",
    "macock_xxxxx_yyyyy_00height_paradiseregaind2ed",
    "newcomb_8675_309_00height_theophania",
    "newcomb_xxxxx_yyyyy_00height_deathsadvantage",
    "newcomb_xxxxx_yyyyy_00height_thelordofhosts",
    "newcomb_xxxxx_yyyyy_00height_truegain",
    "ratcliffe_R22356_usou_2_fourthpart1669",
    "rhodgkinson_xxxxx_yyyyy_00height_articles",
    "rhodgkinson_xxxxx_yyyyy_00height_commission",
    "rhodgkinson_xxxxx_yyyyy_00height_episcopacy",
    "rroberts_aaaaa_bbbbb_00height_divineprovidence",
    "rroberts_aaaaa_bbbbb_00height_expositionofchurch",
    "rroberts_aaaaa_bbbbb_00height_kingsroyal",
]

folio3 = [
    "anon_R30560_usppic_2_shakespeare3folio1664",
    "daniel_R12045_ncaotu_8_opusculavaria1658",
    "daniel_R226090_uk_2_holywarre1647",
    "hayes_R35661_spmauc_4_deratione1664",
    "hayes_R8679_de12_4_opticapromota1663",
    "hodgkinson_R34604_4_uscu_natureofprocurations1661",
    "hodgkinson_R7102_4_usnnc_restauranda1662",
    "jhayes_R216914_uk_8_aconsolatorydiscourse1665",
    "jhayes_R217129_uk_8_abriefexhortation1665",
    "leach_R217001_uk_4_thehistoryoftarquin1669",
    "ratcliffe_R11702_usnjpt_4_treatiseminister1670",
    "ratcliffe_R203411_uscua_4_meansandmethod1660",
    "ratcliffe_R618_gbuklw_4_sermonpreached1666",
    "tratcliffe_R20427_uk_4_thepastoraloffice1663",
    "tratcliffe_R22215_uk_4_thepeaceofjerusalem1659",
    "tratcliffe_R5572_uk_4_romeinherfruits1663",
    "tratcliffenthompson_R212964_uk_4_greatbritainsglory1672",
    "warren_R15965_uk_4_constitutionsandcanons1662",
    "warren_R230847_uk_4_summeandsubstance1661",
    # missing_books.txt: accidentally skipped these in first alignment run. TODO: reverse applying saved models...
    "leach_xxxxx_yyyyy_00height_acoppy",
    "ratcliffe_R22356_usou_2_fourthpart1669",
    "rhodgkinson_xxxxx_yyyyy_00height_articles",
    "rhodgkinson_xxxxx_yyyyy_00height_commission",
    "rhodgkinson_xxxxx_yyyyy_00height_episcopacy",
]

folio4 = [
    "amaxwell_R153_uk_4_englishschoolmaster1673",
    "amaxwell_R8130_uk_2_choiceandpracticalexpositions1675",
    "anon_R25621_usppic_2_shakespeare4folio1685",
    "everingham_R3105_de12_4_ogygia1685",
    "everingham_R771_de12_2_someyearstravels1677",
    "macock_R10405_ustxhr_4_manofmode1684",
    "macock_R25438_usnnut_4_gentiles1677",
    "macock_R25855_usmiu_4_ephemerides1672",
    "macock_xxxx_de12_4_parsonswedding1663",
    "maxwellroberts_R10994_usnnut_4_gentiles1677",
    "maxwellroberts_R24847_usmb_4_scornfullady1677",
    "rawlins_R23326_uscstmogri_2_musaeumregalis1681",
    # missing_books.txt: accidentally skipped these in first alignment run. TODO: reverse applying saved models...
    "macock_xxxxx_yyyyy_00height_diodorus",
    "macock_xxxxx_yyyyy_00height_paradiseregaind2ed",
    "newcomb_8675_309_00height_theophania",
    "newcomb_xxxxx_yyyyy_00height_deathsadvantage",
    "newcomb_xxxxx_yyyyy_00height_thelordofhosts",
    "newcomb_xxxxx_yyyyy_00height_truegain",
    "rroberts_aaaaa_bbbbb_00height_divineprovidence",
    "rroberts_aaaaa_bbbbb_00height_expositionofchurch",
    "rroberts_aaaaa_bbbbb_00height_kingsroyal",    # TODO: WRONG ROBERTS! EXCLUDE!
]

folio4part1 = [
    "amaxwellrroberts_R5112_bl_4_theimprovableness1681",
    "maxwellroberts_R10994_usnnut_4_gentiles1677",
    "maxwellroberts_R11482_bl_2_magnaveritas1680",
    "maxwellroberts_R24847_usmb_4_scornfullady1677",
    "rroberts_R34831_bl_4_thehistoryofconformity1689",
]
folio4part1.extend([
    f"anon_R25621_usppic_2_shakespeare4folio1685-{page_num:04d}" for page_num in range(293)
])

folio4part2 = [
    "amaxwellrroberts_R5112_bl_4_theimprovableness1681",
    "maxwellroberts_R10994_usnnut_4_gentiles1677",
    "maxwellroberts_R11482_bl_2_magnaveritas1680",
    "maxwellroberts_R24847_usmb_4_scornfullady1677",
    "rroberts_R34831_bl_4_thehistoryofconformity1689",
]
folio4part2.extend([
    f"anon_R25621_usppic_2_shakespeare4folio1685-{page_num:04d}" for page_num in range(293)
])

folio4part3= [
    "amaxwellrroberts_R5112_bl_4_theimprovableness1681",
    "maxwellroberts_R10994_usnnut_4_gentiles1677",
    "maxwellroberts_R11482_bl_2_magnaveritas1680",
    "maxwellroberts_R24847_usmb_4_scornfullady1677",
    "rroberts_R34831_bl_4_thehistoryofconformity1689",
]
folio4part3.extend([
    f"anon_R25621_usppic_2_shakespeare4folio1685-{page_num:04d}" for page_num in range(293)
])

def check_filename_in_set(file_name, folio_number):
    if folio_number == '3':
        book_names = folio3
    elif folio_number == '4':
        book_names = folio4
    elif folio_number == '4p1':
        book_names = folio4part1

    for book_name in book_names:
        if book_name in file_name:
            return True
    return False


if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folio_number', type=str, choices=['3', '4', '4p1'])
    args = parser.parse_args()

    for file_name in sys.stdin:
        if check_filename_in_set(file_name.strip(), args.folio_number):
            print(file_name.strip())
