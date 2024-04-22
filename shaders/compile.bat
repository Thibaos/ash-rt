glslang --target-env spirv1.5 -V -o %~dp0\spv\rgen.spv %~dp0\rt.rgen
@REM spirv-as --target-env spv1.5 %~dp0\spv\rgen.spvasm -o %~dp0\spv\rgen.spv 
@REM spirv-val %~dp0\spv\rgen.spv 

glslang --target-env spirv1.5 -V -o %~dp0\spv\rchit.spv %~dp0\rt.rchit
@REM spirv-as --target-env spv1.5 %~dp0\spv\rchit.spvasm -o %~dp0\spv\rchit.spv 
@REM spirv-val %~dp0\spv\rchit.spv 

glslang --target-env spirv1.5 -V -o %~dp0\spv\rmiss.spv %~dp0\rt.rmiss
@REM spirv-as --target-env spv1.5 %~dp0\spv\rmiss.spvasm -o %~dp0\spv\rmiss.spv 
@REM spirv-val %~dp0\spv\rmiss.spv 

glslang --target-env spirv1.5 -V -o %~dp0\spv\rint.spv %~dp0\rt.rint 
@REM spirv-as --target-env spv1.5 %~dp0\spv\rint.spvasm -o %~dp0\spv\rint.spv 
@REM spirv-val %~dp0\spv\rint.spv 

@REM del %~dp0\spv\*.spvasm

exit /b ERRORLEVEL