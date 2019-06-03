function concept(a::Int, b::Int, c::Int)

    for s in 1:a
        global val = 15
        global x
        global flag = true
        println("S-Schleife ",s)
        for k in 1:b
            d = 1
            α = 0.5
            β = 2.0
            println("K-Schleife ", k)
            while flag
                x = α * β * d * c
                println("While Schleife")
                if x <=(val)
                    println("If Bedingung")
                    val = x
                    d = 1
                    break
                else
                    println("Else Bedingung")
                    d = d * β
                end

            end
        end
    end
end

concept(10,2,2)
