glslang --target-env spirv1.5 -V -o %~dp0\spv\rt_rgen.spv %~dp0\rt.rgen
if errorlevel 1 exit /b %ERRORLEVEL%
glslang --target-env spirv1.5 -V -o %~dp0\spv\rt_rchit.spv %~dp0\rt.rchit
if errorlevel 1 exit /b %ERRORLEVEL%
glslang --target-env spirv1.5 -V -o %~dp0\spv\rt_rmiss.spv %~dp0\rt.rmiss
if errorlevel 1 exit /b %ERRORLEVEL%
glslang --target-env spirv1.5 -V -o %~dp0\spv\rt_rint.spv %~dp0\rt.rint 
if errorlevel 1 exit /b %ERRORLEVEL%