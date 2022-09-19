require 'image'
trainsize=0.8
testsize=0.2
local data =  '../videos/IJBC_128_96_new/GT/'

local dirs = dir.getdirectories(data);

file = io.open("train.txt", "a")
file2 = io.open("test.txt", "a")

classess={} --video class
allvideos={}
for k,dirpath in ipairs(dirs) do
	-- print('k,dirpath',k,dirpath)
	-- her klasorde birden fazla video olabilir.
	-- bunlari da almak lazim
	-- bunlardan bazılarını train ve test yapmak lazım
	-- class - videos şeklinde dizidir.
	classess[k]=dirpath
	-- bir sıralı k indexine sınıfları aldım. klasör yolu şeklinde
	-- print(classess[k])
end

sayac=0
for k=1,#classess do
	diric = dir.getdirectories(classess[k]);
	for i,dirpath in ipairs(diric) do
		sayac=sayac+1
		--print(diric)
		allvideos[sayac]=dirpath
		--print(allvideos[sayac])
	end
end
print(sayac)
-- tüm klasörlerdeki videolar alındı

-- random %80 train seti
math.randomseed(30)
local function shuffleTable( t )
 
	if ( type(t) ~= "table" ) then
		print( "WARNING: shuffleTable() function expects a table" )
		return false
	end
 
	local j
 
	for i = #t, 2, -1 do
		j = math.random( i )
		t[i], t[j] = t[j], t[i]
	end
	return t
end

lastdizi=shuffleTable(allvideos)
for k=1,#lastdizi do
	--print(lastdizi[k])
end

--print(math.ceil(#lastdizi*0.8))
print(#lastdizi)
testSet, trainSet = table.splice(lastdizi, 1, math.ceil(trainsize*#lastdizi))
--print(#trainSet)
--print(#testSet)

for k=1,#trainSet do
	--print(trainSet[k])
	file:write(trainSet[k], "\n")
end

for k=1,#testSet do
	--print(trainSet[k])
	file2:write(testSet[k], "\n")
end

-- closes the open file
file:close()
file2:close()
print(trainSet)
