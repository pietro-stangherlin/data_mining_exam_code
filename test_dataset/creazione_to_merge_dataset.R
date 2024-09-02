# Questo dataset è costruito con l'obbiettivo di essere successivamente
# unito al dataset principale tramite la chiave x1

# chiave
x_id = 0:100

# nuove variabili 
x10 = rnorm(x_id)

x11 = ifelse(x10 > 50, "a", "b")

df = data.frame(x_id = x_id,
                x10 = x10,
                x11 = x11)

str(df)


write.csv(df, "df_merge.csv", row.names = TRUE, na = "")
