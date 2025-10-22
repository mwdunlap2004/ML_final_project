library(sqldf)
library('tidyverse')
library(dplyr)
soil_temp = read.csv("Soil_Temperature.csv")
air_pressure = read.csv("Air_Pressure.csv")
air_temp = read.csv("Air_Temperature.csv")
humidity = read.csv("Relative_Humidity.csv")
solar_irradiance = read.csv("Solar_Irradiance.csv")
tree_locations = read.csv("Tree_locations.csv")
tree_trunk = read.csv("Tree_Trunk_Radius.csv")
water_content = read.csv("Volumetric_Water_Content.csv")

soil_temp[,c('yr', 'mo', 'da', 'wk')] <- cbind(year(as.Date(soil_temp$time)),
                                               month(as.Date(soil_temp$time)),
                                               day(as.Date(soil_temp$time)),
                                               week(as.Date(soil_temp$time)))

combined_data <- sqldf("
    WITH daily_ok AS (
        SELECT 
            tru.tree,
            sol.da,
            sol.mo,
            sol.yr,
            SUM(CASE WHEN sol.flag IS NOT NULL AND sol.flag <> '' THEN 1 ELSE 0 END)*1.0/COUNT(sol.flag) AS frac_flag_sol,
            SUM(CASE WHEN airp.flag IS NOT NULL AND airp.flag <> '' THEN 1 ELSE 0 END)*1.0/COUNT(airp.flag) AS frac_flag_airp,
            SUM(CASE WHEN hum.flag IS NOT NULL AND hum.flag <> '' THEN 1 ELSE 0 END)*1.0/COUNT(hum.flag) AS frac_flag_hum
           FROM soil_temp AS sol
        LEFT JOIN tree_trunk AS tru ON sol.tree = tru.tree AND sol.time = tru.time
        LEFT JOIN air_pressure AS airp ON sol.tree = airp.tree AND airp.time = tru.time
        LEFT JOIN humidity AS hum ON sol.tree = hum.tree AND hum.time = tru.time
        GROUP BY tru.tree, sol.da, sol.mo, sol.yr
        HAVING 
            (frac_flag_sol <= 0.25 OR frac_flag_sol IS NULL) AND
            (frac_flag_airp <= 0.25 OR frac_flag_airp IS NULL) AND
            (frac_flag_hum <= 0.25 OR frac_flag_hum IS NULL)
    )

    SELECT 
        tru.site,
        tru.tree,
        tru.plot,
        tre.species,
        MIN(DATE(tru.time)) AS 'Date',
        tre.longitude,
        tre.latitude,
        sol.da,
        sol.mo,
        sol.wk,
        sol.yr,
        AVG(tru.stem_radius) AS 'average_stem_radius',
        (MAX(tru.stem_radius) - MIN(tru.stem_radius)) AS 'change_stem_radius',
        AVG(tru.basal_area) AS 'average_basal_area',
        (MAX(tru.basal_area) - MIN(tru.basal_area)) AS 'change_basal_area',
        AVG(sol.soil_temperature_degC) AS 'average_soil_temperature',
        AVG(airp.air_pressure_kPa) AS 'average_air_pressure',
        AVG(hum.relative_humidity) AS 'average_humidity'
    FROM soil_temp AS sol
    LEFT JOIN tree_locations AS tre ON sol.tree = tre.tree
    LEFT JOIN tree_trunk AS tru ON sol.tree = tru.tree AND sol.time = tru.time
    LEFT JOIN air_pressure AS airp ON sol.tree = airp.tree AND airp.time = tru.time
    LEFT JOIN humidity AS hum ON sol.tree = hum.tree AND hum.time = tru.time
    INNER JOIN daily_ok AS ok
        ON tru.tree = ok.tree 
        AND sol.da = ok.da 
        AND sol.mo = ok.mo 
        AND sol.yr = ok.yr
    WHERE 
        (sol.flag IS NULL OR sol.flag = '') AND
        (airp.flag IS NULL OR airp.flag = '') AND
        (hum.flag IS NULL OR hum.flag = '') 
    GROUP BY tru.tree, sol.da, sol.mo, sol.yr
")
plot_env <- sqldf("
    WITH filtered AS (
        SELECT plot, time, solar_W_per_sqm, VWC_m3_per_m3,
               CASE WHEN (solar_flag IS NOT NULL AND solar_flag <> '') THEN 1 ELSE 0 END AS solar_bad,
               CASE WHEN (water_flag IS NOT NULL AND water_flag <> '') THEN 1 ELSE 0 END AS water_bad
        FROM (
            SELECT plot, time, solar_W_per_sqm, flag AS solar_flag, NULL AS VWC_m3_per_m3, NULL AS water_flag FROM solar_irradiance
            UNION ALL
            SELECT plot, time, NULL AS solar_W_per_sqm, NULL AS solar_flag, VWC_m3_per_m3, flag AS water_flag FROM water_content
        )
    )
    SELECT 
        plot,
        DATE(time) AS date,
        AVG(solar_W_per_sqm) AS avg_solar_irradiance,
        AVG(VWC_m3_per_m3) AS avg_soil_water_content
    FROM filtered
    GROUP BY plot, DATE(time)
    HAVING 
        SUM(solar_bad)*1.0/COUNT(solar_bad) <= 0.25 OR
        SUM(water_bad)*1.0/COUNT(water_bad) <= 0.25
")
final_data <- sqldf("
    SELECT 
        c.*,
        p.avg_solar_irradiance,
        p.avg_soil_water_content
    FROM combined_data AS c
    LEFT JOIN plot_env AS p
        ON c.plot = p.plot
        AND c.Date = p.date
")
rm(soil_temp,air_pressure,air_temp,humidity,solar_irradiance,tree_locations,tree_trunk,water_content)
final_data <- subset(final_data, Date >= "2017-07-05")
write.csv(combined_data,"combined_data.csv")  
write.csv(final_data,"final_data.csv")  
                    
                    
                    
                    
                    
                    
                    
