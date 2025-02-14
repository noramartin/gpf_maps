using DelimitedFiles

####################################
############### RNA ################
####################################
### RBP:

all_TFs = ["'a2bp1'", "'B52'", "'bru-3'", "'CG2931'", "'CG2950'","'CG7903'","'cpo'","'elav'","'how'",
    "'msi'","'lark'","'Hrb27C'","'Hrb87F'","'Hrb98DE'","'orb2'","'pAbp'","'Rbp1'","'Rbp1-like'","'Rbp9'","'rin'","'Rox8'","'SF2'","'sm'","'snf'","'Srp54'","'Sxl'", "'U2af50'"]

gp =Dict()

for tf in all_TFs
    
   
        
        try 
            data = readdlm("/Users/msrivast/Desktop/genonets_output/rbp/Drosophila_melanogaster/tau350/$(tf[2:end-1])_genotype_measures.txt",'\t')
           #println("connected network")
            for i in collect(2:1:size(data)[1])
                cc = "'"*tf[2:end-1]*"'"
                cc = vcat(cc,split(chop(data[i,6][2:end-1]; head=1, tail=1), ", "))
                cc[2] ="'"*cc[2]
                cc[end]=cc[end]*"'"
                seq = data[i,1]
                
             ##### DNA polymerase ###########
                for kk in 1:1:7
                    if seq[kk]=='U'
                        #println(seq)
                        ##### converting the U base to T so that the same code can be used for RNA and DNA
                        seq=seq[1:kk-1]*'T'*seq[kk+1:end]
                        #println(seq)
                    end
                end
               
                gp[seq] = cc
            end
            
            
        catch
        end
    end
    print(gp)


# Assuming gp is some dictionary or data structure holding the genotype-phenotype map
phenotypes = unique(values(gp))
int_pheno = Dict{Any, Int64}()
global k = 1
for p in phenotypes
    int_pheno[p] = k
    global k += 1
end

K = 4
L = 7
# Create an 8-dimensional array for storing the genotype-phenotype map
GPmap = zeros(Int64, K, K, K, K, K, K, K)

#GPmap = GPmap


# Mapping nucleotides to integers (A -> 1, C -> 2, G -> 3, U -> 4)
nucleotide_map = Dict('A' => 1, 'C' => 2, 'G' => 3, 'T' => 4)

a1 = 0
for s1 in ['A', 'C', 'G', 'T']
    a1 = nucleotide_map[s1]
    for s2 in ['A', 'C', 'G', 'T']
        a2 = nucleotide_map[s2]
        for s3 in ['A', 'C', 'G', 'T']
            a3 = nucleotide_map[s3]
            for s4 in ['A', 'C', 'G', 'T']
                a4 = nucleotide_map[s4]
                for s5 in ['A', 'C', 'G', 'T']
                    a5 = nucleotide_map[s5]
                    for s6 in ['A', 'C', 'G', 'T']
                        a6 = nucleotide_map[s6]
                        for s7 in ['A', 'C', 'G', 'T']
                            a7 = nucleotide_map[s7]
                          
                                
                                # Here you can access the corresponding sequence using s1*s2*...s8
                                # Assuming `gp` is some map where sequences like "ACGT..." are keys
                                sequence = string(s1, s2, s3, s4, s5, s6, s7)
                                
                                # Check if this sequence exists in gp and update GPmap if it does
                                if haskey(gp, sequence)
                                    GPmap[a1, a2, a3, a4, a5, a6, a7] = int_pheno[gp[sequence]]
                                    #RC = rc([a1, a2, a3, a4, a5, a6, a7]) 
                                    #GPmap[RC[1], RC[2], RC[3], RC[4], RC[5], RC[6], RC[7]] = int_pheno[gp[sequence]]
                                
                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end


using NPZ
npzwrite("GPmap_RNA.npy", GPmap)


println("done writing file")
