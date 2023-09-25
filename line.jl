using CSV
using Plots
using DataFrames
using Statistics
using LinearAlgebra

data = CSV.File("iris.csv") |> DataFrame

function plot_lda_boundary(data, x_col, y_col, class1, class2)
    class1_data = filter(row -> row.class .== class1, data)
    class2_data = filter(row -> row.class .== class2, data)

    x_data_class1 = class1_data[:, x_col]
    y_data_class1 = class1_data[:, y_col]

    x_data_class2 = class2_data[:, x_col]
    y_data_class2 = class2_data[:, y_col]

    μ1 = [mean(x_data_class1), mean(y_data_class1)]
    Σ1 = cov([x_data_class1 y_data_class1]) + 0.01 * I  

    μ2 = [mean(x_data_class2), mean(y_data_class2)]
    Σ2 = cov([x_data_class2 y_data_class2]) + 0.01 * I 

    Σ_inv = inv(Σ1 + Σ2)
    w = Σ_inv * (μ1 - μ2)

    b = -0.5 * (μ1' * Σ_inv * μ1 - μ2' * Σ_inv * μ2)

    x_minimal = minimum(data[:, x_col]) - 1
    x_maksimal = maximum(data[:, x_col]) + 1
    x_rentang = x_minimal:0.01:x_maksimal

    y_rentang = -(w[1] * x_rentang .+ b) ./ w[2]

    plot(x_data_class1, y_data_class1, seriestype=:scatter, label=class1, shape=:circle)
    plot!(x_data_class2, y_data_class2, seriestype=:scatter, label=class2, shape=:diamond)
    plot!(x_rentang, y_rentang, label="Decision Boundary", linewidth=2)
    xlabel!(string(x_col))
    ylabel!(string(y_col))
    title!("Class Boundary")

    # Menyimpan Plot dalam file .png
    savefig("class_boundary.png")
    
    println("\nHasil Plot telah berhasil disimpan di file class_boundary.png")
end

# Menampilkan Kolom yang tersedia
println("Kolom yang tersedia:")
println(names(data, Not(:class)))

# Input Dinamis untuk memilih kolom yang diinginkan
println("\nMasukkan pilihan kolom yang anda inginkan untuk garis X-axis (dari list kolom diatas): ")
x_column = Symbol(readline())
println("\nMasukkan pilihan kolom yang anda inginkan untuk garis Y-axis (dari list kolom diatas): ")
y_column = Symbol(readline())
println("========================================================================================")

# Menampilkan Class yang tersedia
println("\nClass yang tersedia:")
class_tersedia = unique(data.class)
for (i, class_name) in enumerate(class_tersedia)
    println("[$i] $class_name")
end

# Input Dinamis untuk memilih class yang diinginkan
println("\nMasukkan Angka untuk Class pertama yang anda inginkan dari Class yang tersedia diatas: ")
class1_index = parse(Int, readline())
println("\nMasukkan Angka untuk Class kedua yang anda inginkan dari Class yang tersedia diatas: ")
class2_index = parse(Int, readline())
println("========================================================================================")

# Mengambil nama kelas yang dipilih berdasarkan indeks
class1 = class_tersedia[class1_index]
class2 = class_tersedia[class2_index]

# Memanggil fungsi Plot batas keputusan (decision boundary) LDA
plot_lda_boundary(data, x_column, y_column, class1, class2)
display(plot)

