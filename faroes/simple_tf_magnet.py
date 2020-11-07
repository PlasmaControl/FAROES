#!/usr/bin/env python
# coding: utf-8

# In[55]:


import openmdao.api as om


# In[56]:


import numpy as np
from scipy.constants import mu_0

class InnerTFCoilTension(om.ExplicitComponent):
    
    def setup(self):
        self.add_input('I_leg', units='MA', desc='Current in one leg')
        self.add_input('B0', units='T', desc='Field on axis')
        self.add_input('R0', units='m', desc='Major radius')
        self.add_input('r1', units='m', desc='avg radius of conductor inner leg, at midplane')
        self.add_input('r2', units='m', desc='avg radius of conductor outer leg, at midplane')
        self.add_output('T1', units='MN', desc='Tension on the inner leg')
        
    def compute(self, inputs, outputs):
        i_leg = inputs['I_leg']
        b0    = inputs['B0']
        R0    = inputs['R0']
        r1    = inputs['r1']
        r2    = inputs['r2']
        
        k = np.log(r2/r1)
        T1 = 0.5 * i_leg * b0 * R0 * (r1 + r2 * (k -1)) / (r2 - r1)
        outputs['T1'] = T1      
            
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')


# In[57]:


class FieldAtRadius(om.ExplicitComponent):
    
    def setup(self):
        self.add_input('I_leg', units='MA')
        self.add_input('r_om', units='m')
        self.add_input('R0', units='m')
        self.add_input('n_coil')

        self.add_output('B_on_coil', units='T')
        self.add_output('B0', units='T')
        
    def field_at_radius(self, i, r):
        """
        i: total current, A
        r: meters
        
        Assumes circular symmetry
        
        B = μ0 I / (2 π r)
        """
        b = mu_0 * i / (2 * np.pi * r)
        return b
        
    def compute(self, inputs, outputs):
        n_coil = inputs['n_coil']
        I_leg = inputs['I_leg']
        R_coil_max    = inputs['r_om']
        R0    = inputs['R0']
        
        i_total_MA = 1e6 * n_coil * I_leg

        B_on_coil     = self.field_at_radius(i_total_MA, R_coil_max)
        B0            = self.field_at_radius(i_total_MA, R0)

        outputs['B_on_coil'] = B_on_coil
        outputs['B0'] = B0
        
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')


# In[58]:


class InnerTFCoilStrain(om.ExplicitComponent):
    E_rat = 1.25714
    hts_max_stress = 525
    
    def setup(self):
        self.add_input('T1', 0.002, units='MN', desc='Tension on the inner leg')
        self.add_input('A_s', 0.06, units='m**2')
        self.add_input('A_t', 0.012, units='m**2')
        self.add_input('A_m', 0.012, units='m**2', desc='Magnet area')
        self.add_input('f_HTS', 0.76, desc='Fraction of magnet area which is HTS')
        
        self.add_output('s_HTS', 0.066, units='MPa', desc='Strain on the HTS material')
        self.add_output('max_stress_constraint', 0.5, desc='fraction of maximum stress on the HTS')
    
    def compute(self, inputs, outputs):
        A_s = inputs['A_s']
        A_m = inputs['A_m']
        A_t = inputs['A_t']
        f_HTS = inputs['f_HTS']
        T1 = inputs['T1']
        
        sigma_HTS = ((A_s + A_t) * self.E_rat + f_HTS * A_m) / T1
        outputs['s_HTS'] = sigma_HTS
        outputs['max_stress_constraint'] = (self.hts_max_stress - sigma_HTS)/self.hts_max_stress

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')


# In[59]:


class MagnetGeometry(om.ExplicitComponent):
    e_gap = 0.006 # m 
    Δr_t = 0.05 # m
    
    def setup(self):
        self.add_input('r_ot', 0.405, units='m', desc='Magnet inenr leg outer structure radius')
        self.add_input('n_coil', 18, desc='number of coils')
        self.add_input('r_iu', 8.025, units='m', desc='Inner radius of outer TF leg')
        self.add_input('r_im', 0.22, units='m')
        self.add_input('r_is', 0.1, units='m')
        
        self.add_output('r_it', units='m')
        self.add_output('r_os', units='m')
        self.add_output('r_om', units='m')
        
        self.add_output('w_is', 0.1, units='m')
        self.add_output('w_os', units='m')
        self.add_output('w_im', units='m')
        self.add_output('w_om', 0.250, units='m')
        self.add_output('w_it', units='m')
        self.add_output('w_ot', units='m')

        self.add_output('l_is', units='m')
        self.add_output('l_os', units='m')
        self.add_output('l_im', units='m')
        self.add_output('l_om', units='m')
        self.add_output('l_it', units='m')
        self.add_output('l_ot', units='m')
        
        self.add_output('A_s', 0.06, units='m**2')
        self.add_output('A_t', 0.01, units='m**2')
        self.add_output('A_m', 0.1, units='m**2')
        
        self.add_output('r1', 0.8, units='m')
        self.add_output('r2', 8.2, units='m')
        
    def compute(self, inputs, outputs):
        r_is = inputs['r_is']
        r_im = inputs['r_im']        
        r_ot = inputs['r_ot']
        
        r_iu = inputs['r_iu']
        
        n_coil = inputs['n_coil']
        
        r_it = r_ot - self.Δr_t
        r_om = r_it - self.e_gap        
        r_os = r_im - self.e_gap

        outputs['r_it'] = r_it
        outputs['r_os'] = r_os
        outputs['r_om'] = r_om
        
        r_to_w = np.cos(np.pi/n_coil)
        r_to_l = 2 * np.sin(np.pi/n_coil)
             
        w_is = r_is * r_to_w
        w_os = r_os * r_to_w
        w_im = r_im * r_to_w
        w_om = r_om * r_to_w
        w_it = r_it * r_to_w        
        w_ot = r_ot * r_to_w
        
        outputs['w_is'] = w_is
        outputs['w_os'] = w_os
        outputs['w_im'] = w_im
        outputs['w_om'] = w_om
        outputs['w_it'] = w_it
        outputs['w_ot'] = w_ot
        
        l_is = r_is * r_to_l
        l_os = r_os * r_to_l
        l_im = r_im * r_to_l
        l_om = r_om * r_to_l
        l_it = r_it * r_to_l        
        l_ot = r_ot * r_to_l
        
        outputs['l_is'] = l_is
        outputs['l_os'] = l_os
        outputs['l_im'] = l_im
        outputs['l_om'] = l_om
        outputs['l_it'] = l_it
        outputs['l_ot'] = l_ot

        outputs['A_s'] = (w_os - w_is) * (l_os + l_is) / 2
        outputs['A_m'] = (w_om - w_im) * (l_om + l_im) / 2
        outputs['A_t'] = (w_ot - w_it) * (l_ot + l_it) / 2
        
        outputs['r1']= (r_om + r_im)/2
        outputs['r2']= r_iu + (r_ot - r_is)/2
        
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')
                


# In[60]:


class MagnetCurrent(om.ExplicitComponent):
    
    def setup(self):
        self.add_input('A_m', 0.01, units='m**2')
        self.add_input('f_HTS', 0.76)
        self.add_input('j_HTS', 1, units='MA/m**2')
        self.add_output('I_leg', units='MA')
        self.add_output('j_eff_wp_max', units='MA/m**2')
    
    def compute(self, inputs, outputs):
        A_m = inputs['A_m']
        f_HTS = inputs['f_HTS']
        j_HTS = inputs['j_HTS']
        i_leg = A_m * f_HTS * j_HTS
        outputs['I_leg'] = i_leg
        
    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')
        


# In[81]:


class MagnetDesign(om.Group):
    
    def setup(self):
        #self = self.add_subsystem('forward', om.Group(), promotes=['*'])
        self.add_subsystem('geometry', MagnetGeometry(),
                           promotes_inputs=['r_ot', 'n_coil', 'r_iu', 'r_im', 'r_is'],
                           promotes_outputs=['A_s', 'A_t', 'A_m', 'r1', 'r2', 'r_om'])
        self.add_subsystem('current', MagnetCurrent(),
                            promotes_inputs=['A_m', 'f_HTS', 'j_HTS'],
                            promotes_outputs=['I_leg', 'j_eff_wp_max'])
        self.add_subsystem('field', FieldAtRadius(),
                           promotes_inputs=['I_leg', 'r_om', 'R0', 'n_coil'],
                           promotes_outputs=['B_on_coil', 'B0'])
        self.add_subsystem('tension', InnerTFCoilTension(),
                           promotes_inputs=['I_leg', 'r1','r2', 'R0', 'B0'],
                           promotes_outputs=['T1'])
        self.add_subsystem('strain', InnerTFCoilStrain(),
                           promotes_inputs=['T1', 'A_m', 'A_t', 'A_s', 'f_HTS'],
                           promotes_outputs=['s_HTS', 'max_stress_constraint'])
        
        #self.nonlinear_solver = om.NonlinearBlockGS()
        
        self.set_input_defaults('n_coil', 18)
        #self.set_input_defaults('R0', 1)
        self.set_input_defaults('f_HTS', 0.76)
        
        self.add_subsystem('obj_cmp', om.ExecComp('obj = -B0', B0={'units': 'T'}),
                           promotes=['B0', 'obj'])
        #self.add_subsystem('con_cmp1', om.ExecComp('con1 = 525 - s_HTS', s_HTS=100), promotes=['con1', 's_HTS'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = 18 - B_on_coil', B_on_coil=1), promotes=['con2', 'B_on_coil'])
        self.add_subsystem('con_cmp3', om.ExecComp('con3 = A_m * j_eff_wp_max - I_leg',
                                                   A_m={'units': 'm**2'},
                                                   j_eff_wp_max={'units': 'MA/m**2'},
                                                   I_leg={'units': 'MA'}),
                           promotes=['con3', 'A_m', 'j_eff_wp_max', 'I_leg'])
        
       # self.nonlinear_solver = om.NonlinearBlockGS()


# In[110]:


prob = om.Problem()

prob.model = MagnetDesign()

prob.driver = om.ScipyOptimizeDriver()

prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('r_is', lower=0.03, upper=0.4)
prob.model.add_design_var('r_im', lower=0.05, upper=0.5)
prob.model.add_design_var('j_HTS', lower=0, upper=300)

prob.model.add_objective('obj')


# In[111]:


prob.model.add_constraint('max_stress_constraint', upper=1)


# In[112]:


prob.model.add_constraint('con2', lower=0)
prob.model.add_constraint('con3', lower=0)

prob.setup()

prob.set_val('geometry.r_ot', 0.405)
prob.set_val('geometry.r_iu', 8.025)

prob.set_val('current.j_eff_wp_max', 160)  
prob.set_val('R0', 3)

prob.run_driver()


# In[113]:


all_inputs  = prob.model.list_inputs(values=True)
all_outputs = prob.model.list_outputs(values=True)


# In[ ]:




