
alter session set NLS_DATE_FORMAT = "DD.MM.YY";

insert into GUARANT.EXCHANGERATE_STAT
select distinct ex.*, CUR.ISO_CCODE as CURRENCY_FROM_CCODE, CUR1.ISO_CCODE as CURRENCY_TO_CCODE from DWH.EXCHANGERATE_STAT@dmfr_link ex
left join dwh.currency_hdim@dmfr_link cur on ex.currency_from_uk = cur.uk 
left join dwh.currency_hdim@dmfr_link cur1 on ex.currency_to_uk = cur1.uk 
where 
    cur.ISO_CCODE in ('RUR','USD', 'EUR' ) and 
    cur1.ISO_CCODE in ('RUR','USD', 'EUR' ) and
    ex.xratetype_uk=1 and
   ex.value_day = trunc(sysdate - 1);

/* Гарантии */
BEGIN EXECUTE IMMEDIATE 'DROP TABLE DEALGUARANTEE'; 
EXCEPTION WHEN OTHERS THEN IF sqlcode != -0942 THEN RAISE; END IF; END; 
/
create table DEALGUARANTEE
as select /* + DRIVING_SITE (a)*/
sysdate as VALUE_DAY,--дата фомирования выгрузки
a.deal_uk,--UK сделки
a.start_date, --Дата начала сделки
a.end_plan_date,--Плановая дата окончания
a.end_fact_date,--Фактическая дата окончания
dgg.deal_front_ref,--Номер сделки в фронт-офисе
m1.ccode as MODULE_FRONT_CCODE,--Символьный код фронт-офис системы
dgg.deal_back_ref,--Номер сделки в бэк-офисе
m2.ccode as MODULE_BACK_CCODE,--Символьный код бэк-офис системы
cl1.pin as PRINCIPAL_PIN,--ПИН принципала
cl1.name as PRINCIPAL_NAME,--Наименование принципала
cl2.pin as BENEFICIAR_PIN,--ПИН бенефициара
cl2.name as BENEFICIAR_NAME,--Наименование бенефициара
bb.deal_amt as DEAL_START_AMOUNT,--Начальная величина сделки
a.deal_cur_amt,--Текущая величина сделки
c.iso_ccode as DEAL_CURRENCY,--Символьный код валюты сделки
drs.int_rate as COMISSION_RATE,--Величина ставки комиссии
case 
    when d2d.xk is not null then 'Y'
    else 'N'
end as USED_FLAG, --Флаг раскрытия (через связь сделка-сделка)
case 
    when d2d.xk is not null then hvd.deal_cur_amt
    else null
end as USED_AMOUNT, --Величина раскрытия (величина связанной сделки)
case 
    when d2d.xk is not null then cur1.iso_ccode
    else null
end as USED_CURRENCY,--Валюта раскрытия (валюта связанной сделки)
case 
    when d2a12.xk is not null then 'Y'      
    else 'N'
end as DELAY_FLAG,  --Признак просрочки (через связь счет-сделка)  
p.ccode as product_ccode,--Символьный код продкта
p.name as product_name,--Наименование продукта
dc.class_ccode as dealclass_ccode,--Символьный код класса сделки
dc.name as dealclass_name,--Наименование класса сделки
dk.ccode as dealkind_ccode,--Символьный код разновидности сделки
dk.name as dealkind_name,--Наименование разновидности сделки
ss.BRANCH_EQ_CCODE,--Символьный код филиала
ss.name as SALESPLACE_NAME, --Наименование филиала
PCC.ENG_CCODE as PROFITCENTER_ENG_CCODE, --Код профит-центра (англ.)
PCC.NAME as PROFITCENTER_NAME,--Наименование профит-центра
LP.CCODE as LOANPURPOSE_CCODE,--Символьнный код цели кредита
lp.name as LOANPURPOSE_NAME,--Наименование цели кредита
adv.ccode as ADVREPAYRIGHT_CCODE, --Символьный код типа права досрочного по погашения
adv.name as ADVREPAYRIGHT_NAME  --Наименование типа права досрочного погашения
from dmfr.deal_vhist@dmfr_link a
left join dmfr.dealguarantee_vhist@dmfr_link dgg on 1=1
                                       and a.deal_uk=dgg.deal_uk
                                       and dgg.deleted_flag!='Y'
                                       and dgg.effective_to=to_date('31.12.5999','dd.mm.yyyy')
left join dmfr.module_ldim@dmfr_link m1 on 1=1 
                              and m1.deleted_flag!='Y'
                              and m1.uk=dgg.module_front_uk
left join dmfr.module_ldim@dmfr_link m2 on 1=1 
                              and m2.deleted_flag!='Y'
                              and m2.uk=dgg.module_back_uk                              
/*валюта сделки*/
left join dmfr.currency_sdim@dmfr_link c on 1=1
                               and c.uk=a.currency_uk
                               and c.deleted_flag!='Y'
/*продукт сделки*/                               
left join dmfr.product_sdim@dmfr_link p on 1=1
                               and p.uk=a.product_uk
                               and p.deleted_flag!='Y'
/*класс сделки*/                               
left join dmfr.dealclass_ldim@dmfr_link dc on 1=1
                               and dc.uk=a.dealclass_uk
                               and dc.deleted_flag!='Y'
/*разновидность сделки*/                               
left join dmfr.dealkind_ldim@dmfr_link dk on 1=1
                               and dk.uk=a.dealkind_uk
                               and dk.deleted_flag!='Y'
/*филиал*/                               
left join dmfr.salesplace_sdim@dmfr_link ss on 1=1
                               and ss.uk=a.salesplace_branch_uk
                               and ss.deleted_flag!='Y' 
/*изначальная величина сделки*/                                 
left join 
        (select  * from (
                         select /* + DRIVING_SITE (dgh)*/ 
                         row_number () over (partition by dgh.uk order by dgh.validfrom asc) as rn,dgh.* from dwh.dealguarantee_hdim@dmfr_link dgh
                         where 1=1
                         and dgh.deleted_flag!='Y'
                         and dgh.uk>1
                         and (dgh.dealtype_uk=30 or dgh.uk=6485142622) 
                                         )  where rn=1)  bb  on 1=1
                                                            and a.deal_uk=bb.uk  
/*выход на сделку платежа по гарантии*/                                              
left join dmfr.deal2deal_vhist@dmfr_link d2d on 1=1
                                   and d2d.deleted_flag!='Y'
                                   and a.deal_uk=d2d.deal_main_uk
                                   and d2d.DEAL2DEALTYPE_UK=38 /*Связь Гарантии и Платежа по Гарантии*/
/*сделка платежа по гарантии*/
left join dmfr.deal_vhist@dmfr_link hvd on 1=1
                              and hvd.deleted_flag!='Y'
                              and d2d.deal_related_uk=hvd.deal_uk
                              and hvd.effective_to=to_date('31.12.5999','dd.mm.yyyy')
/*валюта сделки платежа по гарантии*/                              
left join dmfr.currency_sdim@dmfr_link cur1 on 1=1
                                  and cur1.deleted_flag!='Y'
                                  and cur1.uk=hvd.currency_uk
/*выход на бенефициара*/  
left join dwh.dealguarantee_hdim@dmfr_link dg1 on 1=1 
                                       and a.deal_uk=dg1.uk
                                       and dg1.deleted_flag!='Y'
                                       and dg1.validto=to_date('31.12.5999','dd.mm.yyyy')
/*вывод принципала*/                                         
left join dmfr.client_sdim@dmfr_link cl1 on 1=1
                               and a.client_uk=cl1.uk
                               and cl1.deleted_flag!='Y'
/*вывод бенефициара*/                                                                                                                              
left join dmfr.client_sdim@dmfr_link cl2 on 1=1
                               and dg1.CLIENT_BENEFICIARY_UK=cl2.uk
                               and cl2.deleted_flag!='Y'     
/*выведение флага просрочки*/                                                            
left join (select * from  (
                           select /* + DRIVING_SITE (d2a1)*/ 
                           row_number() over (partition by d2a1.deal_uk order by effective_from asc) as RN, 
                           d2a1.* from dmfr.deal2acct_vhist@dmfr_link d2a1
                           where 1=1
                           and d2a1.deleted_flag!='Y'
                           and d2a1.dealtype_uk=30
                           and d2a1.DEAL2ACCTTYPE_UK=3 
                                          ) where rn=1) d2a12 on 1=1
                                              and d2a12.deal_uk=a.deal_uk                                                                                                                                                                     
left join dmfr.dealrate_shist@dmfr_link drs on 1=1
                                 and drs.deleted_flag!='Y'
                                 and drs.effective_to=to_date('31.12.5999','dd.mm.yyyy')
                                 and drs.ratetype_uk=27
                                 and drs.deal_uk=a.deal_uk
left join dmfr.profitcenter_ldim@dmfr_link pcc on 1=1
                                     and a.profitcenter_uk=PCC.UK
                                     and pcc.deleted_flag!='Y'        
left join DMFR.LOANPURPOSE_SDIM@dmfr_link lp on 1=1
                                   and bb.LOANPURPOSE_UK=lp.uk
                                   and lp.deleted_flag!='Y'
left join DWH.ADVREPAYRIGHT_LOV@dmfr_link adv on 1=1
                                    and bb.ADVREPAYRIGHT_UK=ADV.UK                                                                      
where 1=1
and a.deleted_flag!='Y'
and a.effective_to=to_date('31.12.5999','dd.mm.yyyy')
and a.dealtype_uk=30
and bb.deal_amt is not null
order by 2;
commit;

EXIT